# UTM Specification (MVP)

## Purpose
UTMs standardize campaign attribution. Inconsistent UTMs fragment reporting.

## Required parameters (when tagging campaigns)
- `utm_source` (e.g., `google`, `instagram`, `newsletter`)
- `utm_medium` (e.g., `cpc`, `paid_social`, `email`)
- `utm_campaign` (e.g., `summer_sale_2026`)

## Optional parameters
- `utm_content` (creative / ad variation)
- `utm_term` (keyword)

## Formatting rules
- Lowercase only
- Use `snake_case` (underscore) for multi-word values
- Avoid spaces and special characters
- Keep names stable over time

## Allowed values (MVP)
### utm_medium
- `cpc`
- `paid_social`
- `organic_social`
- `email`
- `affiliate`
- `referral`
- `organic_search`
- `direct`

### utm_source examples
- `google`
- `instagram`
- `tiktok`
- `facebook`
- `newsletter`
- `partner_site`

## Example URL
`https://example.com/product/sku123?utm_source=instagram&utm_medium=paid_social&utm_campaign=summer_sale_2026&utm_content=story_ad_1`