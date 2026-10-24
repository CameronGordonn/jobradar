with src as (
    select
        *,
        payload -> 'MatchedObjectDescriptor' as d,
        payload -> 'MatchedObjectDescriptor' -> 'UserArea' -> 'Details' as details,
        payload -> 'MatchedObjectDescriptor' -> 'PositionRemuneration' -> 0 as pay
    from {{ source('jobradar', 'raw_usajobs') }}
)

select
    'usajobs:' || source_id as job_id,
    'usajobs' as source,
    source_id as source_job_id,
    board,
    d ->> 'OrganizationName' as company,
    d ->> 'PositionTitle' as title,
    d ->> 'PositionLocationDisplay' as location,
    d ->> 'PositionURI' as url,
    d ->> 'DepartmentName' as department,
    null::varchar as employment_type,
    case when lower(details ->> 'RemoteIndicator') = 'true' then 'remote' end as workplace_type,
    lower(details ->> 'RemoteIndicator') = 'true' as remote_flag,
    -- USAJOBS dates carry no offset (US Eastern); kept as given.
    try_cast(d ->> 'PublicationStartDate' as timestamp) as posted_at,
    try_cast(d ->> 'ApplicationCloseDate' as timestamp) as closes_at,
    try_cast(pay ->> 'MinimumRange' as double) as salary_min,
    try_cast(pay ->> 'MaximumRange' as double) as salary_max,
    'USD' as salary_currency,
    case pay ->> 'RateIntervalCode'
        when 'PA' then 'year' when 'PM' then 'month' when 'BW' then 'biweekly'
        when 'PW' then 'week' when 'PD' then 'day' when 'PH' then 'hour'
    end as salary_interval,
    first_ingested_at,
    ingested_at,
    run_id
from src
