# Tracking QA Checklist (MVP)

## Event correctness
- [ ] All event names match the approved taxonomy (no casing variants)
- [ ] `event_ts` is populated and in UTC
- [ ] `user_id` and `session_id` are always present
- [ ] `event_id` is unique per event row

## Funnel integrity
- [ ] `view_item` contains `item_id`
- [ ] `add_to_cart` contains `item_id` and `quantity`
- [ ] `start_checkout` contains `cart_value` and `items_count`
- [ ] `purchase` contains `order_id`, `revenue`, `currency`, `items_count`
- [ ] `purchase.revenue` > 0
- [ ] `purchase.order_id` is unique

## UTM governance
- [ ] UTMs are lowercase
- [ ] `utm_medium` is one of the allowed values
- [ ] Campaign links include at least `utm_source`, `utm_medium`, `utm_campaign`

## Experiment fields
- [ ] If `experiment_id` is present, `variant` must be present
- [ ] `assignment_date` <= `event_date` for experiment-tagged events