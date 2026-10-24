{% for m in ['stg_usajobs', 'stg_greenhouse', 'stg_lever', 'stg_ashby'] %}
select * from {{ ref(m) }}
{% if not loop.last %}union all by name{% endif %}
{% endfor %}
