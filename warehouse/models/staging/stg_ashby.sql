with src as (
    select * from {{ source('jobradar', 'raw_ashby') }}
),

pay as (
    select
        source_id,
        min((c ->> 'minValue')::double) as salary_min,
        max((c ->> 'maxValue')::double) as salary_max,
        any_value(c ->> 'currencyCode') as salary_currency,
        any_value(c ->> 'interval') as ashby_interval
    from src, unnest(from_json(payload -> 'compensation' -> 'summaryComponents', '["json"]')) as t (c)
    where c ->> 'compensationType' = 'Salary'
    group by 1
)

select
    'ashby:' || src.source_id as job_id,
    'ashby' as source,
    src.source_id as source_job_id,
    src.board,
    coalesce(src.board_name, src.board) as company,
    src.payload ->> 'title' as title,
    concat_ws(
        '; ', nullif(src.payload ->> 'location', ''),
        nullif(array_to_string(list_transform(
            from_json(src.payload -> 'secondaryLocations', '["json"]'),
            lambda x: x ->> 'location'  -- noqa: RF02
        ), '; '), '')
    ) as location,
    src.payload ->> 'jobUrl' as url,
    src.payload ->> 'department' as department,
    src.payload ->> 'employmentType' as employment_type,
    case lower(src.payload ->> 'workplaceType')
        when 'remote' then 'remote' when 'hybrid' then 'hybrid' when 'onsite' then 'onsite'
    end as workplace_type,
    (src.payload ->> 'isRemote')::boolean as remote_flag,
    {{ utc("src.payload ->> 'publishedAt'") }} as posted_at,
    null::timestamp as closes_at,
    pay.salary_min,
    pay.salary_max,
    pay.salary_currency,
    case pay.ashby_interval
        when '1 YEAR' then 'year' when '1 MONTH' then 'month' when '1 WEEK' then 'week'
        when '1 DAY' then 'day' when '1 HOUR' then 'hour'
    end as salary_interval,
    src.first_ingested_at,
    src.ingested_at,
    src.run_id
from src
left join pay using (source_id)
