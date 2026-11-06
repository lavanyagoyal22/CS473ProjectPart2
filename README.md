# SQL Homework Solutions & Grading Data

A collection of SQL homework solutions (HW1–HW4) written for SQLite, the two databases they run against, and a CSV of student query submissions with manual grades.

## Contents

| File | Description |
|------|-------------|
| `instacart_stu.db` | SQLite database of Instacart grocery orders (used in HW1, HW2) |
| `spy.db` | SQLite database of spy agencies, agents, teams, and missions (used in HW3, HW4) |
| `homework1_solutions.txt` | HW1: Instacart basics, 15 questions |
| `homework2_solutions.sql` | HW2: Instacart subqueries and aggregation, 14 questions |
| `homework3_solutions.txt` | HW3: spy database joins and set operations, 15 questions |
| `homework4_solutions.sql` | HW4: spy database subqueries and analytics, 15 questions |
| `training_data_Part2.csv` | 174 graded student query submissions across HW1–HW4 |

Each solution file lists the question, the expected output, and a reference query.

## Databases

### `instacart_stu.db`
| Table | Columns |
|-------|---------|
| `product` | product_id, product_name, aisle_id, department_id |
| `order-product` | order_id, product_id, add_to_cart_order, reordered |
| `department` | department_id, department |
| `aisle` | aisle_id, aisle |
| `order` | order_id, user_id, order_number, order_dow, order_hour_of_day, days_since_prior_order |

Note: `order` and `order-product` are reserved or hyphenated names, so quote them (`"order"`, `` `order-product` ``).

### `spy.db`
`agent`, `affiliation` / `affiliationrel`, `language` / `languagerel`, `skill` / `skillrel`, `team` / `teamrel`, `mission`, `securityclearance`.
The `*rel` tables are many-to-many link tables.

## Topics by Homework

- **HW1 (Instacart):** `GROUP BY` / `HAVING`, `LIKE` filters, inner and left joins, `DISTINCT`, self-joins, `INTERSECT` / `EXCEPT`, anti-joins, product co-occurrence.
- **HW2 (Instacart):** scalar and correlated subqueries, `MAX` / `MIN` per group, ties, busiest/least-busy analysis, reorder statistics, cart-order analysis.
- **HW3 (Spy):** `NATURAL JOIN`, multi-table `INNER` / `LEFT` joins, `UNION` / `INTERSECT`, string concatenation, aggregation with `HAVING`.
- **HW4 (Spy):** salary extremes, team averages vs. global averages, rare skills, shared agents between teams, dominant language per group, double agents, mission success/failure counts.

## Training Data (`training_data_Part2.csv`)

Student-submitted queries with grades, likely intended for building or evaluating an automated SQL grader.

| Column | Meaning |
|--------|---------|
| `id` | Row ID |
| `hw_num` | Homework number (1–4) |
| `question_number` | Question within the homework |
| `question_prompt` | The question text |
| `q` | Student's SQL query |
| `r` | Result or output of the query |
| `manual_grade` | Grade assigned by a human (numeric, e.g. 55–100) |
| `manual_explanation` | Grader's explanation |
| `db_instance_path` | Database the query ran on (`./DB/spy.db` or `./DB/instacart_stu.db`) |

Rows per homework: HW1 = 39, HW2 = 44, HW3 = 47, HW4 = 44.

## Running the Queries

```bash
sqlite3 instacart_stu.db
sqlite> SELECT order_dow, COUNT(*) AS order_count FROM "order" GROUP BY order_dow;
```

Or from Python:

```python
import sqlite3
con = sqlite3.connect("spy.db")
print(con.execute("SELECT COUNT(*) FROM agent").fetchone())
```

To load the grading data:

```python
import pandas as pd
df = pd.read_csv("training_data_Part2.csv")
```

## Notes

- All queries target SQLite syntax (backtick and double-quote identifiers, `||` for concatenation).
- Some homework questions have more than one valid solution (e.g., HW1 Q15 uses both `EXCEPT` and a `LEFT JOIN ... IS NULL` approach).
