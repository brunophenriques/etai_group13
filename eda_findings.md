# EDA — Complete Key Findings for the Classification Task

**Project:** Everything & Then Some — *Will They Return, and How Much Will They Spend?*  
**Current scope:** `repeat_purchase_90d` (classification only)  
**Purpose:** Share the important exploratory findings with the team before preprocessing and model development.  
**Evidence:** Checked the saved code and outputs across all **102 cells** of `eda(1).ipynb` (87 code, 15 Markdown). The final section on `clientes.csv` and rolling activity windows also incorporates the additional results shared in our conversation; those analyses are **not contained in this uploaded notebook**. This is a synthesis of outputs, not a rerun of the original CSVs.

> **How to read the numbers:** Except where explicitly indicated, percentages are **observation-level** repeat purchase rates, not the percentage of distinct customers who returned. A customer may contribute several purchase occasions. Associations here are descriptive; no model has established independent feature importance or causation.

## 1. The dataset and target

- `train.csv` contains **6,534 rows, 40 columns and 610 distinct customers**. There are **no duplicated complete rows**, **no duplicated `ID` values**, and **no missing `customer_id` values**.
- The target `repeat_purchase_90d` has **3,654 positives (55.92%)** and **2,880 negatives (44.08%)**. The classification labels are reasonably balanced; a majority-class baseline would be ~55.9% accuracy, although accuracy alone will not be sufficient to evaluate models.
- Nonmissing `purchase_date` values span **11 July 2015 to 15 February 2025**. The rows are repeated *purchase occasions*, not independent customers.
- The separate regression target, `spend_next_90d`, is known for **3,425** rows and missing for **3,109**. It is missing for **all 2,880 non-returners**, plus **229 returners**. This missingness is target-related, not ordinary missing predictor data. **Do not use `spend_next_90d` as a classification input.**
- The notebook initially looked at the regression target: among its **3,425 observed positive values**, median spending was **985.74**, mean **1,921.53**, maximum **25,654.33**, indicating a right-skewed distribution. The full regression EDA remains a separate task.

## 2. Data quality: findings we cannot ignore

### 2.1 A block of almost completely empty observations

- **65 rows (about 1.0%)** have every transaction and historical feature missing. Only `ID`, `customer_id` and `repeat_purchase_90d` remain.
- They belong to **53 distinct customers**, with **32 negative** and **33 positive** classification labels. Hence there is no single-class explanation for their missingness.
- Most affected customers have other complete observations; a few have only incomplete observations in training. However, **the missing purchase date and historical context cannot safely be reconstructed from another row**.
- The notebook tested customer-based modes for some categorical features. This looks promising for stable fields but is **not a blanket solution**: a mode calculated using future records is leakage, and a customer mode does not recover the unknown date or transaction values.

### 2.2 Missing historical values: expected vs unexpected

- `prior_mean_gap_days` has **1,308** missing entries; `prior_std_gap_days` has **1,478**.
- The logic of the features explains much of this: **603 observations** have 0 previous purchases (no historical purchase gaps); **445** have 1 previous purchase (still no interval *between previous purchases*); **365** have 2 previous purchases (one interval, insufficient for a sample standard deviation). These are structurally missing values, not necessarily data errors.
- However, **195 observations** have a missing `prior_mean_gap_days` **despite at least 2 prior purchases**. Some have a nonmissing `prior_std_gap_days`. This is an additional data-quality issue, separate from the structural missingness.
- **16 observations** have `prior_mean_gap_days = -4`, an implausible negative historical gap. The notebook found this exact sentinel-like value in multiple customers and years; its origin remains unconfirmed. Do not silently interpret it as a real number of days.
- The customer mean purchase value is sometimes missing even when a prior purchase exists: approximately **4.5% of observations with one previous purchase** lack `prior_mean_purchase_value`. A purely structural rule does not explain every missing value.
- `spend_365d`, `payment_type`, `salesperson`, `warehouse` and `transport_method` have roughly **259–261** missing values each, compared with **65** for most transaction fields. There are therefore additional partial-missingness patterns beyond the 65 almost-empty rows.

### 2.3 Financial fields are internally consistent, but not necessarily complete

- The identity **`Total Bruto − Total Desc_ Linha − Total Desc_ Global = Total Liquido`** holds across the **6,469 complete transaction rows** to floating-point precision (maximum absolute residual ~`3.64e-12`). The simpler identity without the global discount does **not** hold universally.
- `Total Desc_ Global` is **zero in 6,359/6,469** complete rows and positive in **110**; line discounts are much more common (**4,028 positive**, **2,441 zero**). No negative discounts were observed.
- An additional follow-up check found **486 transactions with `Custo = 0` but `Total a Pagar > 0`**, with median payable amount **229.755** and maximum **11,031.74**. These may indicate unrecorded costs, but the business explanation is not established. Do not equate zero cost with zero revenue.
- Follow-up checks found `Total Portes = 0` for **5,416** complete transactions. These zeros may reflect ordinary business practices rather than missingness.

## 3. The strongest recurring theme: customer history matters

### 3.1 Returning vs non-returning purchase occasions

Median features by target in the notebook:

| Historical measure | Did not return (0) | Returned (1) |
|---|---:|---:|
| `prior_purchase_count` | 6 | **17** |
| `days_since_previous_purchase` | 105 days | **37 days** |
| `prior_total_spend` | 3,246.53 | **12,098.99** |
| `purchase_count_90d` | 0 | **1** |
| `spend_90d` | 0 | **746.19** |
| `customer_tenure_days` | 781 | **1,032** |

**Interpretation:** Observations followed by another purchase generally come from customers who bought more often, bought more recently, and had spent more historically. These variables are not independent of one another.

### 3.2 Prior purchase frequency: exceptionally clear monotonic pattern

| Number of previous purchases | Observations | Repeat rate |
|---|---:|---:|
| 0 | 603 | **33.33%** |
| 1 | 445 | **37.30%** |
| 2–5 | 1,202 | **42.35%** |
| 6–10 | 959 | **48.28%** |
| 11–20 | 1,185 | **56.20%** |
| 21–50 | 1,367 | **69.93%** |
| 51+ | 708 | **93.22%** |

This is one of the clearest findings in the notebook. It is a descriptive association, not proof that making a customer purchase once more *causes* their retention probability to rise.

### 3.3 Recency: long-inactive customers have much lower observed repeat rates

| Time since previous purchase | Observations | Repeat rate |
|---|---:|---:|
| 0–7 days | 702 | **79.49%** |
| 8–30 days | 1,355 | **69.08%** |
| 31–60 days | 832 | **68.99%** |
| 61–90 days | 564 | **66.13%** |
| 91–180 days | 1,430 | **41.68%** |
| 181–365 days | 547 | **36.93%** |
| 365+ days | 227 | **29.07%** |

The clearest drop occurs when the last purchase was **more than 90 days** ago. Recency is unavailable for customers without a known previous purchase and must be treated appropriately.

### 3.4 Recency × frequency: the combination is more informative

- In the **0–30-day recency** band, the repeat rate rises from **49.2%** for **0–5 previous purchases** to **95.7%** for **51+ previous purchases** (notebook heatmap and commentary).
- Across the combined recency/frequency heatmap, customers with many historical purchases and recent activity are consistently the most likely to return, while less-established or long-inactive groups have lower observed rates.
- **Sparse cells matter:** the 51+ purchases / 180+ days-since-last-purchase cell has only **3 observations**; do not make strong claims from its rate.
- The analysis suggests testing interactions/nonlinear patterns later, **not** assuming two variables are interchangeable.

### 3.5 Recent rolling-window activity: particularly compelling follow-up result

*The following counts were obtained in the follow-up EDA discussed in chat, rather than in the uploaded notebook's saved cells.*

| Purchases in the preceding 90 days | Observations | Repeat rate |
|---|---:|---:|
| 0 | 2,889 | **37.90%** |
| 1 | 1,714 | **56.77%** |
| 2 | 787 | **71.16%** |
| 3–4 | 477 | **84.70%** |
| 5–9 | 375 | **96.53%** |
| 10+ | 227 | **100.00%** |

- The distinction **0 vs ≥1 prior purchase in the preceding 90 days** was **37.9% vs ~70.6%**.
- Analogous comparisons across the **30-, 180-, and 365-day windows** also showed higher repeat rates with recent activity: approximately **47.8% vs 72.6%** (30d), **33.8% vs 62.1%** (180d), and **32.0% vs 59.6%** (365d).
- The striking 100% rate in the 10+ group is an observed property of these 227 rows, **not a future guarantee**. Those rows may reflect a small cluster of highly engaged customers. Confirm all rolling features are computed strictly before each purchase occasion.

## 4. Historical spending is associated with retention — but strongly overlaps with frequency

| `prior_total_spend` quintile | Observations | Repeat rate |
|---|---:|---:|
| €0–956.87 | 1,294 | **34.93%** |
| €956.87–4,149.69 | 1,294 | **44.44%** |
| €4,149.69–10,611.24 | 1,293 | **53.44%** |
| €10,611.24–28,047.77 | 1,294 | **65.61%** |
| €28,047.77–247,865.10 | 1,294 | **81.45%** |

- This is a strong, monotonic relationship at face value.
- Crucially, `prior_purchase_count` and `prior_total_spend` have a **Spearman correlation of 0.90327**: customers who buy more often also generally accumulate more spending.
- Average **value per previous purchase**, analysed separately for customers with purchase history, has **non-monotonic** quintile repeat rates: **57.9%, 46.1%, 56.6%, 67.6%, 63.3%**.
- **Takeaway:** Historical *frequency* may help explain the strong association between total historical spend and retention. The notebook does **not** establish which feature has greater independent predictive value after controlling for the other.

## 5. Current purchase size and complexity are weaker univariate patterns

### 5.1 Amount payable on the current purchase

The repeat rates for `Total a Pagar` quintiles are **56.80%, 50.93%, 52.28%, 55.95%, and 63.91%**, from the smallest to largest purchase-value group (each approximately 1,294 observations).

- The top-value group does better, but the progression is **not monotonic**.
- Current purchase value gives a noticeably less consistent descriptive signal than prior purchase frequency, recency and total historical spending.

### 5.2 Number of details (`N_ Detalhes`)

| Details in purchase | Observations | Repeat rate |
|---|---:|---:|
| 1 | 720 | **60.97%** |
| 2–5 | 1,408 | **60.01%** |
| 6–10 | 1,026 | **52.83%** |
| 11–20 | 1,112 | **50.72%** |
| 21–30 | 691 | **53.11%** |
| 31+ | 1,512 | **57.14%** |

- Median number of details: **11**; maximum **423**. There is no simple “larger order ⇒ more likely to return” relationship.
- Invoice count and other transaction fields were examined at a descriptive level, but the notebook provides no compelling independently established pattern for them.

## 6. Discounts: large raw associations, substantial customer confounding

### 6.1 Any line discount vs none

| Line discount applied? | Observations | Distinct customers | Median previous purchases | Median recency | Median historical spend | Repeat rate |
|---|---:|---:|---:|---:|---:|---:|
| No | 2,441 | 411 | 7 | 63 days | 4,572.55 | **45.06%** |
| Yes | 4,028 | 469 | 13 | 56 days | 8,599.08 | **62.59%** |

- The raw repeat-rate difference is **17.52 percentage points**.
- However, discounted purchases involve customers with more historical purchases, greater accumulated spend and somewhat more recent activity. **We cannot infer a causal benefit of discounts**.
- Distinct-customer counts **overlap**: the same customer may appear in both discounted and non-discounted rows.

### 6.2 Discount amount and percentage

- Among discounted transactions, increasing **absolute line discount** quintiles have repeat rates **57.37%, 53.86%, 61.91%, 67.83%, 71.96%**. This is generally increasing but not perfectly monotonic.
- Discount *amount* is not a clean measure of discount generosity because it also scales with transaction size.
- The notebook computed `discount_pct = Total Desc_ Linha / Total Bruto × 100` (positive line-discount rows only). Its median is **16.00%**, mean **17.67%**, maximum **97.47%**.
- Discount-percentage quintile repeat rates: **51.49%, 56.89%, 62.16%, 60.25%, 82.13%**. The **top group (>28.88%)** is notably high.
- But that highest-percentage group has **125 distinct customers**, a **median 41.5 previous purchases** and **11-day recency**; the lowest-percentage group has **186 customers**, **15 prior purchases**, and **92-day recency**. The high-discount group also has a **smaller median current transaction value (285.53 vs 945.09)**.
- **Takeaway:** Discounts are associated with repeat purchasing, but they may largely act as markers of established client relationships. An independent effect remains untested.

## 7. Commercial and operational categories: beware of customer concentration

### 7.1 Payment type

- `payment_type` has **9 categories**. The largest, **Payment 7**, contains **4,565 observations** with **52.73%** repeat rate; **Payment 12** has **991** and **54.29%**.
- **Payment 9** appears exceptional: **502 observations**, **95.02%** repeat rate. But those rows belong to only **10 customers**, with **median 109.5 previous purchases**, **8-day recency**, and **39,383.73 historical spend**.
- Other payment categories have very small sample sizes (e.g. 4, 8, 9 or 21 rows). Their apparent differences are unstable.
- **Conclusion:** Payment type may distinguish customer segments; it has **not** been shown to cause retention or add value beyond customer history.

### 7.2 Salesperson

- `salesperson` has **9 categories**. **VLOOKUP Master 1**: **4,054 observations**, **330 customers**, **62.43%** repeat; **VLOOKUP Master 4**: **1,461 observations**, **199 customers**, **39.43%** repeat.
- The profiles differ: Master 1 has median **14 previous purchases / 46-day recency**, while Master 4 has **8 previous purchases / 103-day recency**.
- Some salespeople have only 3–12 observations, so extremes such as 0% or 100% mean very little.
- **Conclusion:** Salesperson-level rates are substantially confounded by different customer portfolios.

### 7.3 Delivery method, zone, warehouse and document series

- `transport_method` is **high cardinality (80 distinct values)**. Delivery 11 has **98.02%** repeat rate across **253 rows**, but just **2 customers**, with **127 previous purchases** and **5-day recency** at the median. This is a textbook customer-concentration effect.
- `commercial_zone` is extremely imbalanced: Zone 4 accounts for **6,260 of 6,469** complete observations (**96.8%**) and has **56.53%** repeat; Zone 2 has **151 / 41.06%**, Zone 3 has **56 / 35.71%**, and two zones have **one observation each**.
- `warehouse` is similarly concentrated: Warehouse 1 has **6,253** observations and **55.65%** repeat; Warehouse 3 has only **21** observations and **80.95%** repeat.
- `document_series` provides almost **no separation** by target: Series 1 **56.13%** (1,828 rows), Series 2 **55.81%** (2,738), Series 3 **56.07%** (1,903).
- **Conclusion:** Rare operational codes need careful treatment; even large row counts can represent very few actual customers. Document series is particularly weak in simple group comparisons.

### 7.4 How stable are categorical values within a customer?

The notebook calculated the **average dominant-category share within customer**:

| Category | Average proportion of that customer's known rows in their most common category |
|---|---:|
| `commercial_zone` | **99.85%** |
| `warehouse` | **99.13%** |
| `payment_type` | **96.05%** |
| `salesperson` | **90.92%** |
| `transport_method` | **74.95%** |
| `document_series` | **66.67%** |

- A separate, stricter calculation among customers with **at least five known observations** found only **19.26%** of customers had one *unchanging* transport method, and **18.08%** had one unchanging document series. By contrast, strict consistency was **98.89% for commercial zone** and **97.01% for warehouse**.
- **Important distinction:** A high dominant-category share does not imply that the category is absolutely constant, and these figures were computed retrospectively using all rows in the notebook.
- **Implication:** Customer-based imputation is more plausible for zone/warehouse than for transport/series, but any operational imputation must be **past-only** and must distinguish unknown customers.

## 8. Time and apparent seasonality

### 8.1 Annual differences

- The train dates cover parts of **2015–2025**. Annual repeat rates vary: **2017: 63.28%** (670 observations), **2022: 48.09%** (628), **2024: 60.24%** (757).
- **2025 has only 78 dated observations**, all from the early part of the year, so it is not comparable with a full year. Similarly, 2015 begins in July.
- There is no obvious universal upward or downward trend across years; changing customer composition may contribute.

### 8.2 Calendar-month peaks

| Month | Observations | Repeat rate | Median recency |
|---|---:|---:|---:|
| March | **876** | **43.84%** | **91 days** |
| July | **874** | **48.86%** | **102 days** |
| November | **884** | **46.83%** | **70 days** |
| January | 444 | **68.92%** | 43 days |
| May | 196 | **72.45%** | 41 days |
| September | 436 | **69.27%** | 42.5 days |

- **March, July and November** show the highest transaction volumes, but **lower repeat rates** than several other months. These peaks are about **four months apart**.
- Their transactions are also associated with **longer customer recency**, particularly March and July. They may partly be purchase occasions from customers with longer reorder cycles.
- In the three peak months, the notebook found **158 customers appearing in all three**, **85 appearing in two**, and **197 appearing in only one**. These values aggregate across multiple years and do **not** establish a fixed three-month/yearly purchasing schedule for individuals.
- The year-by-month heatmap shows considerable variation and some low-count month-year cells. We cannot claim stable seasonality from the pooled monthly averages alone.

### 8.3 A possible four-month repurchase rhythm

- **820 of 5,673** known `days_since_previous_purchase` values (**14.5%**) are between **105 and 135 days** (~4 months).
- **175 of 5,673** (**3.1%**) are between **225 and 255 days** (~8 months).
- Together with the March/July/November peaks, this suggests a **possible subgroup of customers with an approximately four-month purchase cycle**.
- This is **a hypothesis, not a demonstrated periodic process**: different customers contribute unevenly, the calendar is uneven, and a formal within-customer periodicity analysis was not conducted.

## 9. Purchase-gap regularity: surprising result, then explanation

- A coefficient of variation was calculated as `gap_cv = prior_std_gap_days / prior_mean_gap_days`, restricted to **positive mean gaps and available standard deviations**. It was available for **4,870** observations (median ~**0.828**, mean ~**0.841**).
- The initial hypothesis was that **more regular purchasing (smaller CV) predicts more returns**. The unadjusted data did **not** support a monotonic version of that hypothesis.

| `gap_cv` quintile | Observations | Repeat rate | Median mean gap | Median previous purchases | Median recency |
|---|---:|---:|---:|---:|---:|
| Lowest, ≤0.589 | 974 | **44.15%** | 101.79 days | 9 | 98 days |
| 0.589–0.752 | 974 | **53.70%** | 77.76 days | 16 | 75 days |
| 0.752–0.904 | 974 | **63.76%** | 60.32 days | 21 | 49 days |
| 0.904–1.071 | 974 | **76.08%** | 48.34 days | 26 | 28 days |
| Highest, >1.071 | 974 | **68.38%** | 59.85 days | 18 | 33 days |

- At first glance, *more variable* histories seem to correspond to higher retention, up to the fourth quintile.
- But those higher-CV groups also have **shorter typical gaps**, **more purchase history**, and **more recent purchases**. For a 90-day prediction horizon, buying often can matter much more than buying on a perfectly regular schedule.
- A mean-gap × CV heatmap reinforced that **shorter average historical gaps** are strongly associated with higher repeat rates. Some CV differences within similar mean-gap bands remain, but they are **not proof of an independent effect**.
- **Takeaway:** Do not interpret CV in isolation. A customer who purchases irregularly every ~30 days may be more likely to buy within the next 90 days than one who purchases very regularly every ~120 days.

## 10. Customer metadata (`clientes.csv`) — findings from the subsequent chat analyses

> **Source distinction:** The uploaded `eda(1).ipynb` ends with purchase-gap regularity. The findings below were calculated and shared later in this conversation, so they are included for team completeness but are **not independently verifiable from that notebook alone**.

### 10.1 Coverage and structure

- `clientes.csv` contains **1,143 customers**, with **unique `customer_id`**. All customers appearing in train and test have corresponding metadata; a many-to-one join to the training purchase occasions is therefore appropriate.
- The test set contains **986 purchase occasions** from **206 customers**; **184 customers** also appear in train and **22** are new relative to train. The metadata table contains **511 customers absent from both datasets**.
- `customer_type` is **entirely missing**. `company_name` is almost unique (**1,141 distinct values** for 1,143 customers); fine-grained fields such as address, postal code and street are also high-cardinality, potentially making raw encoding unwieldy.
- Missing values within metadata include `commercial_region` (**114/1,143**), and substantial missingness in `island` and `city`. This is not the same as missing customer rows after the join: **coverage is complete, attributes are not**.

### 10.2 Country distribution and observed target rates

- **CT10 contains 1,079/1,143 metadata customers (94.4%)**. On the training purchase occasions, CT10 accounts for **6,321 rows from 572 customers**, with a **56.49%** repeat rate.
- **CT5** has **194 training observations / 30 distinct customers**, repeat rate **39.18%**. The remaining countries have only a handful of training rows. This imbalance makes country comparisons inherently fragile.
- `country_or_region` is therefore dominated by a single country; it is not a source of rich balanced geographical variation in this dataset.

### 10.3 Commercial regions

| Region | Training observations | Unique training customers | Median previous purchases | Median recency | Repeat rate |
|---|---:|---:|---:|---:|---:|
| RG17 | **4,109** | **302** | 12 | 75 days | **51.86%** |
| RG23 | 564 | 33 | 42 | 9 days | **82.45%** |
| RG18 | 368 | 33 | 9.5 | 51 days | **61.96%** |
| RG24 | 235 | 10 | 85.5 | 10 days | **94.47%** |
| RG29 | 214 | 22 | 7 | 70 days | **50.47%** |
| RG31 | 165 | 23 | 6 | 85 days | **46.06%** |

- There are **37 nonmissing commercial-region categories** in metadata. RG17 is dominant, but several others have dozens of distinct customers.
- RG23 and RG24's high repeat rates coincide with **very frequent and recent buying**, with RG24 involving only **10 customers**. There is no evidence here that the region *itself* causes customer loyalty.
- `commercial_region` (metadata) and `commercial_zone` (transaction data) are distinct variables; do not conflate them.

### 10.4 Customer registration cohorts

- `customer_since_year` ranges from **2015 to 2026** in metadata. In the matched training observations, earlier cohorts tend to have stronger buying histories, but rates are **not monotonic**:
  - **2015:** 3,573 observations / 240 customers, **59.53%** repeat; median 14 previous purchases.
  - **2017:** 885 observations / 74 customers, **66.78%** repeat; median 16 previous purchases.
  - **2018:** 467 observations / 46 customers, **34.90%** repeat; median 8 previous purchases.
  - **2022:** 111 observations / 39 customers, **25.23%** repeat; median 1 previous purchase.
  - **2023:** 108 observations / 24 customers, **60.19%** repeat; median 2 previous purchases — a reminder that small cohorts can be unusual.
- `customer_since_year` is **not equivalent to tenure at the transaction date**. The purchase-specific `customer_tenure_days` already exists, and no earlier-than-registration transaction dates were found in the follow-up check.

### 10.5 Geographic missingness is systematic

- Among **CT10** metadata customers: missing `island` **47.6%**, `city` **48.0%**, `commercial_region` **8.8%**.
- Among **CT5**: missing `island` **100%**, `city` **44.9%**, `commercial_region` **16.3%**.
- All other countries also lack `island`, making its missingness plausibly **structural/non-applicable for certain locations**, though the actual geographic coding is unknown.
- City missingness is common in both major countries, while commercial region is substantially more complete. For countries with 1–4 customers, country-specific missing percentages are not reliable evidence of a broad pattern.

## 11. What we can confidently tell the team

1. **The clearest behavioral signal is whether a customer is an established, active buyer:** prior frequency, recency, and rolling purchase counts show strong associations with another purchase within 90 days.
2. **Historical spending is informative descriptively but strongly correlated with frequency.** Large prior spend is not independently established as a separate retention driver.
3. **Discounts, payment methods, delivery codes, salespeople and regions have striking raw rates, but many are explained at least partly by customer mix and concentration.** Do not make causal or “best channel/region” claims.
4. **Current transaction size, item/detail count and document series show weaker or non-monotonic patterns** in this univariate EDA.
5. **Potential purchase-cycle patterns exist** (March/July/November and ~120-day gaps), but within-customer seasonality needs more evidence.
6. **The data require targeted quality handling**, especially 65 nearly blank rows, structurally undefined history, 195 anomalously missing mean gaps, 16 negative mean gaps, and zero recorded costs.
7. **Rows are not independent:** many purchases come from the same customers, and the test period is chronologically later. Leakage prevention and realistic time-aware validation are central to credible model evaluation.

## 12. Specific questions to resolve before fitting classification models

| Issue | Next decision/check | Why it matters |
|---|---|---|
| 65 almost-empty rows | Compare safe exclusion / minimal metadata-only strategy; document the choice | They lack purchase date and history, so normal imputation is not credible |
| Historical missing values | Distinguish “not defined yet” from “unexpectedly absent” | Zero purchases and data errors require different treatments |
| 195 missing mean gaps and 16 `-4` values | Confirm whether they are upstream calculation/sentinel issues | Using them as ordinary numeric values would be misleading |
| Categorical missingness | Test missing category / strictly historical customer-based imputation | Full-customer aggregation can incorporate future rows |
| Rare categorical levels | Use sensible grouping or encodings evaluated on training folds | Avoid memorizing a handful of customers |
| `customer_type` | Exclude, as it is all missing | No usable variation |
| `random_noise`, `ID`, `customer_id`, `company_name` | Treat as IDs/noise or evaluate leakage/generalization risks carefully | Memorization and accidental non-generalizable correlations |
| Financial zero costs | Preserve, flag or inspect definition before changing | Zero may mean unknown rather than genuinely free |
| Rolling historical features | Verify time cutoffs and definitions | Every predictor must have existed by the purchase timestamp |
| Train/validation strategy | Use an appropriate chronological split and customer-aware diagnostics | Random rows can overstate generalization when customers repeat |
| Customer metadata | Join by `customer_id`; select sensible geography / cohort features | Metadata is fully matched but sparse/high-cardinality in places |
| Leakage from targets | Keep `spend_next_90d` out of classification features | The second target contains future information |

**Scope boundary:** This document captures the main quantitative and qualitative findings for **classification EDA** from the saved notebook and the subsequent discussed checks. It deliberately does **not** claim verified predictive importances, proven causal effects, completed preprocessing, or a finished regression EDA.
