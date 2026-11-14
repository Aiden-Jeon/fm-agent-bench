# Will the driver accept the coupon?

## Description

Each row is one answer to a survey question. A respondent is shown a driving scenario (where they are going, who is in the car, the weather, the time of day) together with a coupon for a nearby venue that is sent to the car, and says whether they would accept the coupon. Your task is to predict, for each scenario in the test set, whether the coupon is accepted.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `AcceptCoupon` is `Yes`. The file should contain a header and have the following format:

```
id,AcceptCoupon
4,0.5
9,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 24 features and the target `AcceptCoupon`
- `test.csv` — the test set: `id` and the same features. Predict `AcceptCoupon` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Scenario columns

- `destination` — where the driver is going: `No Urgent Place`, `Home`, `Work`
- `passenger` — who is in the car: `Alone`, `Friend(s)`, `Partner`, `Kid(s)`
- `weather` — `Sunny`, `Rainy`, `Snowy`
- `temperature` — outside temperature in °F (30, 55 or 80)
- `time` — hour of the day (7, 10, 14, 18 or 22)
- `coupon` — type of venue the coupon is for: `Coffee House`, `Restaurant(<20)` (meal under $20 per person), `Restaurant(20-50)` ($20–50 per person), `Carry out & Take away`, `Bar`
- `expiration` — the coupon expires in 2 hours (`2h`) or 1 day (`1d`)
- `toCoupon_GEQ15min` — 1 if driving to the venue takes at least 15 minutes, else 0
- `toCoupon_GEQ25min` — 1 if driving to the venue takes at least 25 minutes, else 0
- `direction_same` — 1 if the venue is in the same direction as the current destination, else 0
- `direction_opp` — 1 if the venue is in the opposite direction, else 0

Respondent columns

- `gender`, `age` (bins: `below21`, `21`, `26`, `31`, `36`, `41`, `46`, `50plus`), `maritalStatus`, `has_children` (0/1), `education`, `occupation`, `income` (yearly income bracket)
- `car` — type of car the respondent drives (missing for almost all rows)
- `Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50` — how many times per month the respondent goes to a bar, a coffee house, gets take-away food, goes to a restaurant under $20 per person, or to a restaurant at $20–50 per person: `never`, `less1` (less than once), `1~3`, `4~8`, `gt8` (more than 8). These can be missing.

Target

- `AcceptCoupon` — `Yes` if the coupon is accepted, `No` otherwise (about 57% `Yes`)

All columns except `temperature` and `time` are categorical (the 0/1 columns are flags). Some binned categories are ordered (age, income, visit frequency).
