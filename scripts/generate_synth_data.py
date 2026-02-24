"""
Synthetic data generator for: Growth Analytics OS (MVP)

Outputs (synthetic):
- data/synthetic/users.parquet
- data/synthetic/events.parquet
- data/synthetic/orders.parquet

Design goals:
- 50k users by default
- ~2,000,000 events by default (scalable)
- Realistic funnel behavior: page_view -> view_item -> add_to_cart -> start_checkout -> purchase
- Retention decay via session day offsets
- Experiment fields: experiment_id, variant, assignment_date (user-level assignment)

NOTE: This is synthetic data for portfolio purposes.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Config / constants
# -----------------------------

COUNTRIES = ["ES", "GB", "DE", "FR", "IT", "NL", "PT", "US", "CO", "MX"]
COUNTRY_WEIGHTS = np.array([0.18, 0.12, 0.12, 0.10, 0.10, 0.06, 0.05, 0.10, 0.10, 0.07])

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = np.array([0.62, 0.33, 0.05])

CHANNELS = [
    "paid_social",
    "organic_search",
    "email",
    "affiliate",
    "referral",
    "direct",
    "organic_social",
]
CHANNEL_WEIGHTS = np.array([0.22, 0.26, 0.10, 0.06, 0.08, 0.20, 0.08])

PAID_SOCIAL_SOURCES = ["instagram", "facebook", "tiktok"]
ORGANIC_SOCIAL_SOURCES = ["instagram", "tiktok", "facebook"]
SEARCH_SOURCES = ["google", "bing"]
EMAIL_SOURCES = ["newsletter"]
AFFILIATE_SOURCES = ["partner_site"]
REFERRAL_SOURCES = ["partner_blog", "review_site"]
DIRECT_SOURCES = [None]

SEARCH_TERMS = ["running shoes", "hoodie", "wireless headphones", "backpack", "water bottle", "jacket", "sneakers"]
ITEM_CATEGORIES = ["shoes", "apparel", "electronics", "accessories", "outdoors"]
PAYMENT_TYPES = ["card", "paypal", "apple_pay", "google_pay"]


@dataclass
class GenerationParams:
    n_users: int
    avg_sessions_per_user: float
    max_sessions_per_user: int
    target_events: int
    start_date: date
    end_date: date
    seed: int
    out_dir: str
    chunk_sessions: int
    experiment_id: str
    experiment_share: float
    treatment_uplift: float  # uplift on purchase probability (synthetic effect)


# -----------------------------
# Helpers
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _random_dates(rng: np.random.Generator, n: int, start: date, end: date) -> np.ndarray:
    """Uniform random dates between start and end (inclusive)."""
    start_ts = int(pd.Timestamp(start).value // 10**9)
    end_ts = int(pd.Timestamp(end).value // 10**9)
    rand_ts = rng.integers(start_ts, end_ts + 1, size=n)
    return pd.to_datetime(rand_ts, unit="s").date


def _channel_to_utms(rng: np.random.Generator, channel: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (utm_source, utm_medium) for a given acquisition channel."""
    if channel == "paid_social":
        return rng.choice(PAID_SOCIAL_SOURCES), "paid_social"
    if channel == "organic_social":
        return rng.choice(ORGANIC_SOCIAL_SOURCES), "organic_social"
    if channel == "organic_search":
        return rng.choice(SEARCH_SOURCES), "organic_search"
    if channel == "email":
        return rng.choice(EMAIL_SOURCES), "email"
    if channel == "affiliate":
        return rng.choice(AFFILIATE_SOURCES), "affiliate"
    if channel == "referral":
        return rng.choice(REFERRAL_SOURCES), "referral"
    if channel == "direct":
        return None, "direct"
    # fallback
    return None, None


def _make_event_id(i: int) -> str:
    return f"evt_{i:09d}"


def _make_session_id(i: int) -> str:
    return f"s_{i:09d}"


def _make_user_id(i: int) -> str:
    return f"u_{i:06d}"


def _make_order_id(i: int) -> str:
    return f"ord_{i:09d}"


# -----------------------------
# Users
# -----------------------------

def generate_users(p: GenerationParams) -> pd.DataFrame:
    rng = np.random.default_rng(p.seed)

    user_ids = [_make_user_id(i) for i in range(1, p.n_users + 1)]
    countries = rng.choice(COUNTRIES, size=p.n_users, p=COUNTRY_WEIGHTS)
    devices = rng.choice(DEVICES, size=p.n_users, p=DEVICE_WEIGHTS)
    channels = rng.choice(CHANNELS, size=p.n_users, p=CHANNEL_WEIGHTS)

    signup_dates = _random_dates(rng, p.n_users, p.start_date, p.end_date)

    utm_source = []
    utm_medium = []
    utm_campaign = []
    for ch in channels:
        src, med = _channel_to_utms(rng, ch)
        utm_source.append(src)
        utm_medium.append(med)
        # simple campaign naming (synthetic)
        if med in (None, "direct"):
            utm_campaign.append(None)
        else:
            utm_campaign.append(rng.choice(["always_on", "brand_push", "promo_drop", "summer_sale_2026"]))

    # Experiment assignment at user-level
    in_exp = rng.random(p.n_users) < p.experiment_share
    variant = np.where(in_exp, rng.choice(["control", "treatment"], size=p.n_users), None)

    # assignment_date: on/after signup, within 0-7 days
    assign_offsets = rng.integers(0, 8, size=p.n_users)
    assignment_date = [
        (sd + timedelta(days=int(off))) if in_exp[i] else None
        for i, (sd, off) in enumerate(zip(signup_dates, assign_offsets))
    ]

    users = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": pd.to_datetime(signup_dates),
        "country": countries,
        "device_category": devices,
        "acquisition_channel": channels,
        "utm_source": utm_source,
        "utm_medium": utm_medium,
        "utm_campaign": utm_campaign,
        "experiment_id": np.where(in_exp, p.experiment_id, None),
        "variant": variant,
        "assignment_date": pd.to_datetime(assignment_date),
        "is_synthetic": True,
    })

    return users


# -----------------------------
# Sessions
# -----------------------------

def generate_sessions(p: GenerationParams, users: pd.DataFrame) -> pd.DataFrame:
    """
    Create sessions with retention decay:
    - sessions_per_user ~ Poisson(avg_sessions_per_user) + 1, capped
    - session_date = signup_date + day_offset where day_offset follows a geometric-like distribution
    """
    rng = np.random.default_rng(p.seed + 1)

    # sessions per user
    sess_counts = rng.poisson(lam=p.avg_sessions_per_user, size=len(users)) + 1
    sess_counts = np.clip(sess_counts, 1, p.max_sessions_per_user)

    total_sessions = int(sess_counts.sum())

    # expand user-level data to session-level arrays
    user_id_rep = np.repeat(users["user_id"].values, sess_counts)
    signup_rep = np.repeat(users["signup_date"].values.astype("datetime64[D]"), sess_counts)

    # retention decay: small offsets more likely (geometric distribution)
    # geometric gives 1..inf, subtract 1 => 0..inf
    day_offsets = rng.geometric(p=0.18, size=total_sessions) - 1
    day_offsets = np.clip(day_offsets, 0, 180)

    session_dates = (signup_rep + day_offsets.astype("timedelta64[D]")).astype("datetime64[D]")
    end_d = np.datetime64(p.end_date)
    session_dates = np.minimum(session_dates, end_d)

    # random time in day
    seconds = rng.integers(0, 24 * 3600, size=total_sessions)
    session_start_ts = (session_dates.astype("datetime64[s]") + seconds.astype("timedelta64[s]"))

    sessions = pd.DataFrame({
        "session_idx": np.arange(1, total_sessions + 1),
        "session_id": [_make_session_id(i) for i in range(1, total_sessions + 1)],
        "user_id": user_id_rep,
        "session_start_ts": pd.to_datetime(session_start_ts),
        "session_date": pd.to_datetime(session_dates),
    })

    return sessions


# -----------------------------
# Events per session (funnel simulation)
# -----------------------------

def _session_funnel_probs(channel: str, device: str) -> Dict[str, float]:
    """
    Base probabilities controlling funnel transitions.
    These are synthetic assumptions (not real business results).
    """
    # baseline
    p_view_item = 0.55
    p_add_to_cart = 0.18
    p_start_checkout = 0.55
    p_purchase = 0.62  # conditional on start_checkout

    # channel tweaks (synthetic)
    if channel in ("paid_social", "organic_social"):
        p_view_item += 0.05
        p_add_to_cart -= 0.02
    if channel == "email":
        p_add_to_cart += 0.03
        p_start_checkout += 0.05
    if channel == "organic_search":
        p_view_item += 0.02
        p_add_to_cart += 0.01
    if channel == "affiliate":
        p_purchase += 0.03
    if channel == "direct":
        p_purchase += 0.02

    # device tweaks (synthetic)
    if device == "mobile":
        p_start_checkout -= 0.03
        p_purchase -= 0.03
    if device == "desktop":
        p_add_to_cart += 0.01
        p_purchase += 0.02

    # clamp
    return {
        "p_view_item": float(np.clip(p_view_item, 0.05, 0.95)),
        "p_add_to_cart": float(np.clip(p_add_to_cart, 0.01, 0.60)),
        "p_start_checkout": float(np.clip(p_start_checkout, 0.05, 0.95)),
        "p_purchase": float(np.clip(p_purchase, 0.05, 0.95)),
    }


def generate_events_and_orders_parquet(
    p: GenerationParams,
    users: pd.DataFrame,
    sessions: pd.DataFrame,
) -> Tuple[str, str]:
    """
    Generate events + orders and write incrementally to Parquet (chunked by sessions).
    Returns (events_path, orders_path).
    """
    rng = np.random.default_rng(p.seed + 2)

    # join needed user attributes for session simulation
    user_map = users.set_index("user_id")[[
        "country", "device_category", "acquisition_channel",
        "utm_source", "utm_medium", "utm_campaign",
        "experiment_id", "variant", "assignment_date"
    ]]

    out_events = os.path.join(p.out_dir, "events.parquet")
    out_orders = os.path.join(p.out_dir, "orders.parquet")

    # Parquet writers (append chunks)
    events_writer = None
    orders_writer = None

    event_counter = 0
    order_counter = 0
    total_sessions = len(sessions)

    # We may want to roughly hit target_events.
    # We'll limit session processing once we reach target_events.
    progress = tqdm(total=total_sessions, desc="Generating sessions")

    for start in range(0, total_sessions, p.chunk_sessions):
        end = min(start + p.chunk_sessions, total_sessions)
        chunk = sessions.iloc[start:end].copy()

        # enrich with user attributes
        attrs = user_map.loc[chunk["user_id"]].reset_index(drop=True)
        chunk = pd.concat([chunk.reset_index(drop=True), attrs], axis=1)

        events_rows: List[Dict] = []
        orders_rows: List[Dict] = []

        for row in chunk.itertuples(index=False):
            # stop if we already generated enough events
            if event_counter >= p.target_events:
                break

            session_start = pd.Timestamp(row.session_start_ts)
            user_id = row.user_id
            session_id = row.session_id
            country = row.country
            device = row.device_category
            channel = row.acquisition_channel

            utm_source = row.utm_source
            utm_medium = row.utm_medium
            utm_campaign = row.utm_campaign

            exp_id = row.experiment_id
            variant = row.variant
            assignment_date = row.assignment_date  # Timestamp or NaT

            probs = _session_funnel_probs(channel, device)

            # uplift: only affects users in treatment (synthetic effect)
            purchase_uplift = p.treatment_uplift if (variant == "treatment") else 0.0
            p_purchase = min(0.99, probs["p_purchase"] * (1.0 + purchase_uplift))

            # helper to attach experiment fields only if event is after assignment_date
            def exp_fields(event_ts: pd.Timestamp) -> Dict:
                if pd.isna(assignment_date) or exp_id is None or variant is None:
                    return {"experiment_id": None, "variant": None, "assignment_date": None}
                # assignment_date is DATE-level; compare by date
                if event_ts.date() >= pd.Timestamp(assignment_date).date():
                    return {"experiment_id": exp_id, "variant": variant, "assignment_date": pd.Timestamp(assignment_date).date()}
                return {"experiment_id": None, "variant": None, "assignment_date": None}

            # 1) page_view (always)
            t0 = session_start
            event_counter += 1
            events_rows.append({
                "event_id": _make_event_id(event_counter),
                "event_ts": t0,
                "event_date": t0.date(),
                "user_id": user_id,
                "session_id": session_id,
                "event_name": "page_view",
                "country": country,
                "device_category": device,
                "acquisition_channel": channel,
                "utm_source": utm_source,
                "utm_medium": utm_medium,
                "utm_campaign": utm_campaign,
                "utm_content": None,
                "utm_term": None,
                "page_type": rng.choice(["home", "category", "product", "search", "cart", "checkout"], p=[0.25, 0.25, 0.25, 0.10, 0.10, 0.05]),
                "page_path": None,
                "referrer": rng.choice([None, "https://google.com", "https://instagram.com", "https://tiktok.com"], p=[0.55, 0.20, 0.15, 0.10]),
                "item_id": None,
                "item_category": None,
                "price": None,
                "search_term": None,
                "results_count": None,
                "click_target": None,
                "click_text": None,
                "quantity": None,
                "cart_value": None,
                "items_count": None,
                "order_id": None,
                "revenue": None,
                "currency": None,
                "payment_type": None,
                "is_synthetic": True,
                **exp_fields(t0),
            })

            # optional search event
            if rng.random() < 0.12:
                dt = int(rng.integers(5, 45))
                t = t0 + pd.Timedelta(seconds=dt)
                event_counter += 1
                term = rng.choice(SEARCH_TERMS)
                events_rows.append({
                    "event_id": _make_event_id(event_counter),
                    "event_ts": t,
                    "event_date": t.date(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_name": "search",
                    "country": country,
                    "device_category": device,
                    "acquisition_channel": channel,
                    "utm_source": utm_source,
                    "utm_medium": utm_medium,
                    "utm_campaign": utm_campaign,
                    "utm_content": None,
                    "utm_term": term.replace(" ", "_"),
                    "page_type": "search",
                    "page_path": None,
                    "referrer": None,
                    "item_id": None,
                    "item_category": None,
                    "price": None,
                    "search_term": term,
                    "results_count": int(rng.integers(0, 120)),
                    "click_target": None,
                    "click_text": None,
                    "quantity": None,
                    "cart_value": None,
                    "items_count": None,
                    "order_id": None,
                    "revenue": None,
                    "currency": None,
                    "payment_type": None,
                    "is_synthetic": True,
                    **exp_fields(t),
                })

            # optional click event
            if rng.random() < 0.20:
                dt = int(rng.integers(10, 60))
                t = t0 + pd.Timedelta(seconds=dt)
                event_counter += 1
                events_rows.append({
                    "event_id": _make_event_id(event_counter),
                    "event_ts": t,
                    "event_date": t.date(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_name": "click",
                    "country": country,
                    "device_category": device,
                    "acquisition_channel": channel,
                    "utm_source": utm_source,
                    "utm_medium": utm_medium,
                    "utm_campaign": utm_campaign,
                    "utm_content": rng.choice([None, "story_ad_1", "banner_a", "banner_b"], p=[0.55, 0.15, 0.15, 0.15]),
                    "utm_term": None,
                    "page_type": None,
                    "page_path": None,
                    "referrer": None,
                    "item_id": None,
                    "item_category": None,
                    "price": None,
                    "search_term": None,
                    "results_count": None,
                    "click_target": rng.choice(["nav_menu", "cta_shop_now", "add_to_cart_button", "promo_banner"]),
                    "click_text": rng.choice([None, "Shop now", "Add to cart", "Learn more"], p=[0.25, 0.25, 0.25, 0.25]),
                    "quantity": None,
                    "cart_value": None,
                    "items_count": None,
                    "order_id": None,
                    "revenue": None,
                    "currency": None,
                    "payment_type": None,
                    "is_synthetic": True,
                    **exp_fields(t),
                })

            # funnel path
            # view_item?
            has_view = rng.random() < probs["p_view_item"]
            if has_view:
                dt = int(rng.integers(20, 120))
                t = t0 + pd.Timedelta(seconds=dt)
                event_counter += 1

                item_id = f"sku{int(rng.integers(1, 5000)):04d}"
                cat = rng.choice(ITEM_CATEGORIES, p=[0.30, 0.28, 0.18, 0.16, 0.08])
                price = float(np.round(rng.uniform(9.0, 220.0), 2))

                events_rows.append({
                    "event_id": _make_event_id(event_counter),
                    "event_ts": t,
                    "event_date": t.date(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_name": "view_item",
                    "country": country,
                    "device_category": device,
                    "acquisition_channel": channel,
                    "utm_source": utm_source,
                    "utm_medium": utm_medium,
                    "utm_campaign": utm_campaign,
                    "utm_content": None,
                    "utm_term": None,
                    "page_type": "product",
                    "page_path": f"/product/{item_id}",
                    "referrer": None,
                    "item_id": item_id,
                    "item_category": cat,
                    "price": price,
                    "search_term": None,
                    "results_count": None,
                    "click_target": None,
                    "click_text": None,
                    "quantity": None,
                    "cart_value": None,
                    "items_count": None,
                    "order_id": None,
                    "revenue": None,
                    "currency": None,
                    "payment_type": None,
                    "is_synthetic": True,
                    **exp_fields(t),
                })

                # add_to_cart?
                has_atc = rng.random() < probs["p_add_to_cart"]
                if has_atc:
                    dt2 = int(rng.integers(10, 90))
                    t2 = t + pd.Timedelta(seconds=dt2)
                    event_counter += 1

                    qty = int(rng.integers(1, 3))

                    events_rows.append({
                        "event_id": _make_event_id(event_counter),
                        "event_ts": t2,
                        "event_date": t2.date(),
                        "user_id": user_id,
                        "session_id": session_id,
                        "event_name": "add_to_cart",
                        "country": country,
                        "device_category": device,
                        "acquisition_channel": channel,
                        "utm_source": utm_source,
                        "utm_medium": utm_medium,
                        "utm_campaign": utm_campaign,
                        "utm_content": None,
                        "utm_term": None,
                        "page_type": "cart",
                        "page_path": "/cart",
                        "referrer": None,
                        "item_id": item_id,
                        "item_category": cat,
                        "price": price,
                        "search_term": None,
                        "results_count": None,
                        "click_target": "add_to_cart_button",
                        "click_text": "Add to cart",
                        "quantity": qty,
                        "cart_value": None,
                        "items_count": None,
                        "order_id": None,
                        "revenue": None,
                        "currency": None,
                        "payment_type": None,
                        "is_synthetic": True,
                        **exp_fields(t2),
                    })

                    # start_checkout?
                    has_checkout = rng.random() < probs["p_start_checkout"]
                    if has_checkout:
                        dt3 = int(rng.integers(20, 120))
                        t3 = t2 + pd.Timedelta(seconds=dt3)
                        event_counter += 1

                        items_count = int(rng.integers(1, 5))
                        cart_value = float(np.round(price * qty * rng.uniform(0.95, 1.15), 2))

                        events_rows.append({
                            "event_id": _make_event_id(event_counter),
                            "event_ts": t3,
                            "event_date": t3.date(),
                            "user_id": user_id,
                            "session_id": session_id,
                            "event_name": "start_checkout",
                            "country": country,
                            "device_category": device,
                            "acquisition_channel": channel,
                            "utm_source": utm_source,
                            "utm_medium": utm_medium,
                            "utm_campaign": utm_campaign,
                            "utm_content": None,
                            "utm_term": None,
                            "page_type": "checkout",
                            "page_path": "/checkout",
                            "referrer": None,
                            "item_id": None,
                            "item_category": None,
                            "price": None,
                            "search_term": None,
                            "results_count": None,
                            "click_target": "checkout_button",
                            "click_text": "Checkout",
                            "quantity": None,
                            "cart_value": cart_value,
                            "items_count": items_count,
                            "order_id": None,
                            "revenue": None,
                            "currency": None,
                            "payment_type": None,
                            "is_synthetic": True,
                            **exp_fields(t3),
                        })

                        # purchase?
                        has_purchase = rng.random() < p_purchase
                        if has_purchase:
                            dt4 = int(rng.integers(30, 180))
                            t4 = t3 + pd.Timedelta(seconds=dt4)
                            event_counter += 1
                            order_counter += 1

                            order_id = _make_order_id(order_counter)
                            revenue = float(np.round(cart_value * rng.uniform(0.98, 1.08), 2))
                            currency = rng.choice(["EUR", "GBP", "USD"], p=[0.70, 0.20, 0.10])
                            pay = rng.choice(PAYMENT_TYPES, p=[0.62, 0.18, 0.10, 0.10])

                            events_rows.append({
                                "event_id": _make_event_id(event_counter),
                                "event_ts": t4,
                                "event_date": t4.date(),
                                "user_id": user_id,
                                "session_id": session_id,
                                "event_name": "purchase",
                                "country": country,
                                "device_category": device,
                                "acquisition_channel": channel,
                                "utm_source": utm_source,
                                "utm_medium": utm_medium,
                                "utm_campaign": utm_campaign,
                                "utm_content": None,
                                "utm_term": None,
                                "page_type": "checkout",
                                "page_path": "/checkout/complete",
                                "referrer": None,
                                "item_id": None,
                                "item_category": None,
                                "price": None,
                                "search_term": None,
                                "results_count": None,
                                "click_target": None,
                                "click_text": None,
                                "quantity": None,
                                "cart_value": cart_value,
                                "items_count": items_count,
                                "order_id": order_id,
                                "revenue": revenue,
                                "currency": currency,
                                "payment_type": pay,
                                "is_synthetic": True,
                                **exp_fields(t4),
                            })

                            # orders table row
                            orders_rows.append({
                                "order_id": order_id,
                                "order_ts": t4,
                                "order_date": t4.date(),
                                "user_id": user_id,
                                "session_id": session_id,
                                "revenue": revenue,
                                "currency": currency,
                                "items_count": items_count,
                                "payment_type": pay,
                                "country": country,
                                "device_category": device,
                                "acquisition_channel": channel,
                                "utm_source": utm_source,
                                "utm_medium": utm_medium,
                                "utm_campaign": utm_campaign,
                                "experiment_id": exp_id if (variant in ("control", "treatment")) else None,
                                "variant": variant if (variant in ("control", "treatment")) else None,
                                "assignment_date": pd.Timestamp(assignment_date).date() if not pd.isna(assignment_date) else None,
                                "is_synthetic": True,
                            })

        # Write chunk to parquet
        if events_rows:
            events_df = pd.DataFrame(events_rows)
            table = pa.Table.from_pandas(events_df, preserve_index=False)

            if events_writer is None:
                events_writer = pq.ParquetWriter(out_events, table.schema, compression="snappy")
            events_writer.write_table(table)

        if orders_rows:
            orders_df = pd.DataFrame(orders_rows)
            table_o = pa.Table.from_pandas(orders_df, preserve_index=False)

            if orders_writer is None:
                orders_writer = pq.ParquetWriter(out_orders, table_o.schema, compression="snappy")
            orders_writer.write_table(table_o)

        progress.update(end - start)

        if event_counter >= p.target_events:
            break

    progress.close()

    if events_writer is not None:
        events_writer.close()
    if orders_writer is not None:
        orders_writer.close()

    return out_events, out_orders


# -----------------------------
# Main
# -----------------------------

def parse_args() -> GenerationParams:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", dest="out_dir", default="data/synthetic", help="Output directory")
    ap.add_argument("--n_users", type=int, default=50_000)
    ap.add_argument("--avg_sessions_per_user", type=float, default=8.0)
    ap.add_argument("--max_sessions_per_user", type=int, default=30)
    ap.add_argument("--target_events", type=int, default=2_000_000)
    ap.add_argument("--start_date", type=str, default="2025-11-01")
    ap.add_argument("--end_date", type=str, default="2026-02-24")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk_sessions", type=int, default=5000)
    ap.add_argument("--experiment_id", type=str, default="exp_checkout_001")
    ap.add_argument("--experiment_share", type=float, default=0.50, help="Share of users in experiment")
    ap.add_argument("--treatment_uplift", type=float, default=0.05, help="Synthetic uplift on purchase probability")

    args = ap.parse_args()

    return GenerationParams(
        n_users=args.n_users,
        avg_sessions_per_user=args.avg_sessions_per_user,
        max_sessions_per_user=args.max_sessions_per_user,
        target_events=args.target_events,
        start_date=pd.to_datetime(args.start_date).date(),
        end_date=pd.to_datetime(args.end_date).date(),
        seed=args.seed,
        out_dir=args.out_dir,
        chunk_sessions=args.chunk_sessions,
        experiment_id=args.experiment_id,
        experiment_share=args.experiment_share,
        treatment_uplift=args.treatment_uplift,
    )


def main() -> None:
    p = parse_args()
    _ensure_dir(p.out_dir)

    print("== Growth Analytics OS: Synthetic Data Generator ==")
    print(f"Output dir: {p.out_dir}")
    print(f"Users: {p.n_users}")
    print(f"Target events: {p.target_events}")
    print(f"Date range: {p.start_date} to {p.end_date}")
    print(f"Experiment: {p.experiment_id} (share={p.experiment_share}, uplift={p.treatment_uplift})")
    print("NOTE: This data is synthetic for portfolio purposes.\n")

    users = generate_users(p)
    sessions = generate_sessions(p, users)

    users_path = os.path.join(p.out_dir, "users.parquet")
    users.to_parquet(users_path, index=False)
    print(f"Wrote: {users_path} (rows={len(users):,})")

    events_path, orders_path = generate_events_and_orders_parquet(p, users, sessions)
    print(f"Wrote: {events_path}")
    print(f"Wrote: {orders_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()