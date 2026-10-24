with src as (
    select * from {{ source('jobradar', 'raw_lever') }}
)

select
    'lever:' || source_id as job_id,
    'lever' as source,
    source_id as source_job_id,
    board,
    coalesce(board_name, board) as company,
    payload ->> 'text' as title,
    coalesce(
        nullif(array_to_string(from_json(payload -> 'categories' -> 'allLocations', '["varchar"]'), '; '), ''),
        payload -> 'categories' ->> 'location'
    ) as location,
    payload ->> 'hostedUrl' as url,
    payload -> 'categories' ->> 'team' as department,
    payload -> 'categories' ->> 'commitment' as employment_type,
    case lower(payload ->> 'workplaceType')
        when 'remote' then 'remote' when 'hybrid' then 'hybrid' when 'onsite' then 'onsite'
    end as workplace_type,
    case when lower(payload ->> 'workplaceType') = 'remote' then true end as remote_flag,
    timezone('UTC', to_timestamp((payload ->> 'createdAt')::bigint / 1000)) as posted_at,
    null::timestamp as closes_at,
    (payload -> 'salaryRange' ->> 'min')::double as salary_min,
    (payload -> 'salaryRange' ->> 'max')::double as salary_max,
    payload -> 'salaryRange' ->> 'currency' as salary_currency,
    case payload -> 'salaryRange' ->> 'interval'
        when 'per-year-salary' then 'year' when 'per-month-salary' then 'month'
        when 'per-week-salary' then 'week' when 'per-day-wage' then 'day' when 'per-hour-wage' then 'hour'
    end as salary_interval,
    first_ingested_at,
    ingested_at,
    run_id
from src
