# Data Dictionary

Source: Kaggle — *Hotel Booking Demand* dataset (originally from Antonio, Almeida & Nunes 2019).
Raw file: `data/raw/hotel_bookings.csv` — **119,390 rows x 32 columns**.

All 29 input features are required in an upload CSV. `is_canceled` is the target;
`reservation_status`, `reservation_status_date` are dropped as leakage features.

| Column | Type | Description | Missing |
|---|---|---|---|
| hotel | str | City Hotel or Resort Hotel | 0 |
| is_canceled | int (0/1) | **Target** — booking cancelled | 0 |
| lead_time | int | Days between booking and arrival | 0 |
| arrival_date_year | int | Year of arrival | 0 |
| arrival_date_month | str | Month of arrival | 0 |
| arrival_date_week_number | int | Week number of arrival | 0 |
| arrival_date_day_of_month | int | Day of month of arrival | 0 |
| stays_in_weekend_nights | int | Weekend nights (Sat/Sun) | 0 |
| stays_in_week_nights | int | Weekday nights (Mon–Fri) | 0 |
| adults | int | Number of adults | 0 |
| children | float | Number of children (0 in most rows) | 4 |
| babies | int | Number of babies | 0 |
| meal | str | Meal plan: BB, HB, SC, FB, Undefined | 0 |
| country | str | Guest country (ISO-3 code) | 487 |
| market_segment | str | Market segment: Direct, Corporate, Online TA, etc. | 0 |
| distribution_channel | str | Booking channel: Direct, TA/TO, GDS, etc. | 0 |
| is_repeated_guest | int (0/1) | Previous guest of hotel | 0 |
| previous_cancellations | int | Prior cancellations by guest | 0 |
| previous_bookings_not_canceled | int | Prior non-cancelled bookings | 0 |
| reserved_room_type | str | Code of reserved room type (A–P) | 0 |
| assigned_room_type | str | Code of assigned room type (A–P) | 0 |
| booking_changes | int | Number of booking changes | 0 |
| deposit_type | str | No Deposit / Non Refund / Refundable | 0 |
| agent | float | Travel agent ID (NA = no agent) | 14435 |
| company | float | Company ID (NA = no company) | 94077 |
| days_in_waiting_list | int | Days on waiting list | 0 |
| customer_type | str | Contract, Group, Transient, Transient-Party | 0 |
| adr | float | Average daily rate (currency) | 0 |
| required_car_parking_spaces | int | Parking spaces requested | 0 |
| total_of_special_requests | int | Number of special requests | 0 |
| reservation_status | str | Canceled / Check-Out / No-Show (**dropped** — leakage) | 0 |
| reservation_status_date | str | Date of status change (**dropped** — leakage) | 0 |

## Cleaning decisions

| Issue | Handling |
|---|---|
| `children` (4 NA) | Impute 0 (consistent with most rows) |
| `country` (487 NA) | Impute `Unknown` |
| `agent` (14435 NA) | Impute `Unknown` |
| `company` (94077 NA, 79%) | Drop column, create `has_company` flag (0/1) |
| `meal` = Undefined | Recode to `SC` |
| Whitespace in strings | Stripped |
| `children` float dtype | Cast to int |
| Duplicate rows (31,994) | Dropped |
| Zero-guest bookings (180) | Dropped (`adults+children+babies == 0`) |
| `adr` typo (5,400) | Dropped rows where `adr >= 5000` |

## Engineered features

Created in the preprocessing pipeline and present in the model input:

| Feature | Definition |
|---|---|
| `total_stay_nights` | `stays_in_weekend_nights + stays_in_week_nights` |
| `is_weekend_only` | 1 if weekend nights only |
| `total_guests` | `adults + children + babies` |
| `is_family` | 1 if children or babies present |
| `is_solo` | 1 if exactly 1 adult and 1 total guest |
| `adr_per_person` | `adr / max(total_guests, 1)` |
| `room_type_changed` | 1 if `reserved_room_type != assigned_room_type` |
| `has_company` | 1 if company ID present |
| `country_freq` | Frequency encoding of `country` (train-derived) |
