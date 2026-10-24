-- Lowercase, punctuation to spaces, common abbreviations spelled out, so that
-- "Sr. ML Engineer" and "Senior Machine Learning Engineer" normalize the same way.
{% macro normalize_title(col) -%}
trim(regexp_replace(
  regexp_replace(regexp_replace(regexp_replace(regexp_replace(regexp_replace(regexp_replace(
    ' ' || regexp_replace(lower({{ col }}), '[^a-z0-9]+', ' ', 'g') || ' ',
    ' sr ', ' senior ', 'g'),
    ' jr ', ' junior ', 'g'),
    ' ml ', ' machine learning ', 'g'),
    ' swe ', ' software engineer ', 'g'),
    ' eng ', ' engineer ', 'g'),
    ' mgr ', ' manager ', 'g'),
  '\s+', ' ', 'g'))
{%- endmacro %}

-- Salary to a yearly figure: 2,080 work hours, 260 work days, 52 weeks, 26 pay periods.
{% macro annualize(amount, interval) -%}
case {{ interval }}
  when 'year' then {{ amount }}
  when 'month' then {{ amount }} * 12
  when 'biweekly' then {{ amount }} * 26
  when 'week' then {{ amount }} * 52
  when 'day' then {{ amount }} * 260
  when 'hour' then {{ amount }} * 2080
end
{%- endmacro %}

-- ISO timestamp with an offset, to naive UTC.
{% macro utc(expr) -%}
timezone('UTC', try_cast({{ expr }} as timestamptz))
{%- endmacro %}
