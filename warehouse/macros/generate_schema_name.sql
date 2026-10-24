-- Use the configured schema name as is (staging, intermediate, marts) instead of
-- dbt's default <target>_<name>.
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name if custom_schema_name else target.schema }}
{%- endmacro %}
