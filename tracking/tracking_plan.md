# Tracking Plan (GA4-style) — Growth Analytics OS (MVP)

**Product:** E-commerce demo store (synthetic data)  
**Goal:** Enable funnel, retention, revenue, and experiment measurement in BigQuery + Tableau.

## Naming conventions
- Event names: `snake_case` (e.g., `add_to_cart`)
- Parameter names: `snake_case`
- Timestamps: `event_ts` in UTC (ISO-8601 in raw exports)
- IDs: `user_id` (synthetic), `session_id`, `event_id`, `order_id`

## Core events (MVP)
| Event name | When it fires | Key parameters |
|---|---|---|
| `page_view` | Page is viewed | `page_type`, `page_path`, `referrer`, `utm_*` |
| `view_item` | Product detail viewed | `item_id`, `item_category`, `price` |
| `search` | User performs a search | `search_term`, `results_count` |
| `click` | User clicks key CTA | `click_target`, `click_text` |
| `add_to_cart` | Add item to cart | `item_id`, `quantity`, `price` |
| `start_checkout` | Checkout starts | `cart_value`, `items_count` |
| `purchase` | Order completed | `order_id`, `revenue`, `currency`, `items_count`, `payment_type` |
| `app_open` | App is opened (optional web/app hybrid) | `platform` |

## Experimentation support (MVP)
We support simple A/B assignment at the user level:
- `experiment_id`
- `variant` (e.g., `control`, `treatment`)
- `assignment_date`

Assignment can be logged as:
- Parameter on every event, or
- A dedicated `experiment_assigned` event (optional upgrade)

For MVP, synthetic generator will attach experiment fields to events after assignment.

## Required dimensions (for modeling)
- User: `user_id`, `signup_date`, `country`, `device_category`, `acquisition_channel`
- Event: `event_name`, `event_ts`, `session_id`
- Marketing: `utm_source`, `utm_medium`, `utm_campaign`, `utm_content`, `utm_term`
- Commerce (purchase): `order_id`, `revenue`, `currency`, `items_count`

## Data quality expectations
- `purchase.order_id` must be unique
- `purchase.revenue` > 0
- `event_name` must be one of the approved events
- UTMs must follow `tracking/utm_spec.md`

## Notes for analysts (why this design)
- Events are designed to support a standard e-commerce funnel and can be modeled into `fact_events` and `fact_orders`.
- UTMs are governed to avoid attribution fragmentation (e.g., `Instagram` vs `instagram`).
- Experiment fields enable SRM checks and uplift estimation in downstream notebooks.