with src as (
    select * from {{ source('jobradar', 'raw_greenhouse') }}
),

-- pay_input_ranges can hold several zones; keep USD and span them.
pay as (
    select
        source_id,
        min((r ->> 'min_cents')::double) / 100 as salary_min,
        max((r ->> 'max_cents')::double) / 100 as salary_max,
        bool_or(r ->> 'title' ilike '%hour%') as hourly
    from src, unnest(from_json(payload -> 'pay_input_ranges', '["json"]')) as t(r)
    where r ->> 'currency_type' = 'USD'
    group by 1
)

select
    'greenhouse:' || src.source_id as job_id,
    'greenhouse' as source,
    src.source_id as source_job_id,
    src.board,
    coalesce(src.board_name, src.payload ->> 'company_name', src.board) as company,
    src.payload ->> 'title' as title,
    src.payload -> 'location' ->> 'name' as location,
    src.payload ->> 'absolute_url' as url,
    src.payload -> 'departments' -> 0 ->> 'name' as department,
    null::varchar as employment_type,
    null::varchar as workplace_type,
    null::boolean as remote_flag,
    {{ utc("src.payload ->> 'first_published'") }} as posted_at,
    null::timestamp as closes_at,
    pay.salary_min,
    pay.salary_max,
    case when pay.salary_min is not null then 'USD' end as salary_currency,
    case when pay.salary_min is null then null when pay.hourly then 'hour' else 'year' end as salary_interval,
    src.first_ingested_at,
    src.ingested_at,
    src.run_id
from src
left join pay using (source_id)
