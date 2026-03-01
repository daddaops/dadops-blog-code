-- SQL queries from the blog post: https://dadops.dev/blog/llm-receipt-parser/
-- These queries analyze grocery spending data stored in the receipts/line_items tables.

-- Monthly spending totals
SELECT
    strftime('%Y-%m', receipt_date) AS month,
    COUNT(*) AS trips,
    ROUND(SUM(total), 2) AS spent
FROM receipts
WHERE receipt_date IS NOT NULL
GROUP BY month
ORDER BY month DESC;

-- Most frequently purchased items
SELECT
    name,
    COUNT(*) AS times_bought,
    ROUND(AVG(unit_price), 2) AS avg_price,
    ROUND(MIN(unit_price), 2) AS cheapest,
    ROUND(MAX(unit_price), 2) AS priciest
FROM line_items
WHERE is_discount = 0
GROUP BY name
HAVING times_bought > 1
ORDER BY times_bought DESC
LIMIT 15;

-- Track price of a specific item over time
SELECT
    r.receipt_date,
    r.store_name,
    li.unit_price
FROM line_items li
JOIN receipts r ON li.receipt_id = r.id
WHERE li.name LIKE '%MILK%'
ORDER BY r.receipt_date;

-- Which store is cheapest for an item?
SELECT
    r.store_name,
    ROUND(AVG(li.unit_price), 2) AS avg_price,
    COUNT(*) AS samples
FROM line_items li
JOIN receipts r ON li.receipt_id = r.id
WHERE li.name LIKE '%EGGS%'
GROUP BY r.store_name
ORDER BY avg_price;
