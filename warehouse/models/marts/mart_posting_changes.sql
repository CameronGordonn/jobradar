-- What changed on each posting after it was first seen, from the snapshot history.
with versions as (
    select
        *,
        row_number() over w as version,
        lag(title_clean) over w as prev_title,
        lag(location) over w as prev_location,
        lag(salary_min_annual) over w as prev_salary_min,
        lag(salary_max_annual) over w as prev_salary_max,
        lag(workplace_type) over w as prev_workplace,
        lag(is_open) over w as prev_open
    from {{ ref('snap_postings') }}
    window w as (partition by job_id order by dbt_valid_from)
)

select
    job_id,
    any_value(source) as source,
    arg_max(company, dbt_valid_from) as company,
    arg_max(role_family, dbt_valid_from) as role_family,
    count(*) as versions,
    count(*) filter (where version > 1 and prev_title is distinct from title_clean) as title_changes,
    count(*) filter (where version > 1 and prev_location is distinct from location) as location_changes,
    count(*) filter (where version > 1 and (prev_salary_min is distinct from salary_min_annual
                                            or prev_salary_max is distinct from salary_max_annual)) as salary_changes,
    count(*) filter (where version > 1 and prev_workplace is distinct from workplace_type) as workplace_changes,
    count(*) filter (where version > 1 and prev_open and not is_open) as closures,
    count(*) filter (where version > 1 and not prev_open and is_open) as reopenings,
    min(dbt_valid_from) as first_snapshot_at,
    max(dbt_valid_from) as last_change_at
from versions
group by job_id
