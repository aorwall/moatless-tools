# django__django-14463

| **django/django** | `68ef274bc505cd44f305c03cbf84cf08826200a8` |
| ---- | ---- |
| **No of patches** | 17 |
| **All found context length** | 20543 |
| **Any found context length** | 453 |
| **Avg pos** | 145.41176470588235 |
| **Min pos** | 1 |
| **Max pos** | 144 |
| **Top file pos** | 1 |
| **Missing snippets** | 52 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/commands/inspectdb.py b/django/core/management/commands/inspectdb.py
--- a/django/core/management/commands/inspectdb.py
+++ b/django/core/management/commands/inspectdb.py
@@ -78,18 +78,16 @@ def table2model(table_name):
             )
             yield "from %s import models" % self.db_module
             known_models = []
-            table_info = connection.introspection.get_table_list(cursor)
-
             # Determine types of tables and/or views to be introspected.
             types = {"t"}
             if options["include_partitions"]:
                 types.add("p")
             if options["include_views"]:
                 types.add("v")
+            table_info = connection.introspection.get_table_list(cursor)
+            table_info = {info.name: info for info in table_info if info.type in types}
 
-            for table_name in options["table"] or sorted(
-                info.name for info in table_info if info.type in types
-            ):
+            for table_name in options["table"] or sorted(name for name in table_info):
                 if table_name_filter is not None and callable(table_name_filter):
                     if not table_name_filter(table_name):
                         continue
@@ -232,6 +230,10 @@ def table2model(table_name):
                     if field_type.startswith(("ForeignKey(", "OneToOneField(")):
                         field_desc += ", models.DO_NOTHING"
 
+                    # Add comment.
+                    if connection.features.supports_comments and row.comment:
+                        extra_params["db_comment"] = row.comment
+
                     if extra_params:
                         if not field_desc.endswith("("):
                             field_desc += ", "
@@ -242,14 +244,22 @@ def table2model(table_name):
                     if comment_notes:
                         field_desc += "  # " + " ".join(comment_notes)
                     yield "    %s" % field_desc
-                is_view = any(
-                    info.name == table_name and info.type == "v" for info in table_info
-                )
-                is_partition = any(
-                    info.name == table_name and info.type == "p" for info in table_info
-                )
+                comment = None
+                if info := table_info.get(table_name):
+                    is_view = info.type == "v"
+                    is_partition = info.type == "p"
+                    if connection.features.supports_comments:
+                        comment = info.comment
+                else:
+                    is_view = False
+                    is_partition = False
                 yield from self.get_meta(
-                    table_name, constraints, column_to_field_name, is_view, is_partition
+                    table_name,
+                    constraints,
+                    column_to_field_name,
+                    is_view,
+                    is_partition,
+                    comment,
                 )
 
     def normalize_col_name(self, col_name, used_column_names, is_relation):
@@ -353,7 +363,13 @@ def get_field_type(self, connection, table_name, row):
         return field_type, field_params, field_notes
 
     def get_meta(
-        self, table_name, constraints, column_to_field_name, is_view, is_partition
+        self,
+        table_name,
+        constraints,
+        column_to_field_name,
+        is_view,
+        is_partition,
+        comment,
     ):
         """
         Return a sequence comprising the lines of code necessary
@@ -391,4 +407,6 @@ def get_meta(
         if unique_together:
             tup = "(" + ", ".join(unique_together) + ",)"
             meta += ["        unique_together = %s" % tup]
+        if comment:
+            meta += [f"        db_table_comment = {comment!r}"]
         return meta
diff --git a/django/db/backends/base/features.py b/django/db/backends/base/features.py
--- a/django/db/backends/base/features.py
+++ b/django/db/backends/base/features.py
@@ -334,6 +334,11 @@ class BaseDatabaseFeatures:
     # Does the backend support non-deterministic collations?
     supports_non_deterministic_collations = True
 
+    # Does the backend support column and table comments?
+    supports_comments = False
+    # Does the backend support column comments in ADD COLUMN statements?
+    supports_comments_inline = False
+
     # Does the backend support the logical XOR operator?
     supports_logical_xor = False
 
diff --git a/django/db/backends/base/schema.py b/django/db/backends/base/schema.py
--- a/django/db/backends/base/schema.py
+++ b/django/db/backends/base/schema.py
@@ -141,6 +141,9 @@ class BaseDatabaseSchemaEditor:
 
     sql_delete_procedure = "DROP PROCEDURE %(procedure)s"
 
+    sql_alter_table_comment = "COMMENT ON TABLE %(table)s IS %(comment)s"
+    sql_alter_column_comment = "COMMENT ON COLUMN %(table)s.%(column)s IS %(comment)s"
+
     def __init__(self, connection, collect_sql=False, atomic=True):
         self.connection = connection
         self.collect_sql = collect_sql
@@ -289,6 +292,8 @@ def _iter_column_sql(
         yield column_db_type
         if collation := field_db_params.get("collation"):
             yield self._collate_sql(collation)
+        if self.connection.features.supports_comments_inline and field.db_comment:
+            yield self._comment_sql(field.db_comment)
         # Work out nullability.
         null = field.null
         # Include a default value, if requested.
@@ -445,6 +450,23 @@ def create_model(self, model):
         # definition.
         self.execute(sql, params or None)
 
+        if self.connection.features.supports_comments:
+            # Add table comment.
+            if model._meta.db_table_comment:
+                self.alter_db_table_comment(model, None, model._meta.db_table_comment)
+            # Add column comments.
+            if not self.connection.features.supports_comments_inline:
+                for field in model._meta.local_fields:
+                    if field.db_comment:
+                        field_db_params = field.db_parameters(
+                            connection=self.connection
+                        )
+                        field_type = field_db_params["type"]
+                        self.execute(
+                            *self._alter_column_comment_sql(
+                                model, field, field_type, field.db_comment
+                            )
+                        )
         # Add any field index and index_together's (deferred as SQLite
         # _remake_table needs it).
         self.deferred_sql.extend(self._model_indexes_sql(model))
@@ -614,6 +636,15 @@ def alter_db_table(self, model, old_db_table, new_db_table):
             if isinstance(sql, Statement):
                 sql.rename_table_references(old_db_table, new_db_table)
 
+    def alter_db_table_comment(self, model, old_db_table_comment, new_db_table_comment):
+        self.execute(
+            self.sql_alter_table_comment
+            % {
+                "table": self.quote_name(model._meta.db_table),
+                "comment": self.quote_value(new_db_table_comment or ""),
+            }
+        )
+
     def alter_db_tablespace(self, model, old_db_tablespace, new_db_tablespace):
         """Move a model's table between tablespaces."""
         self.execute(
@@ -693,6 +724,18 @@ def add_field(self, model, field):
                 "changes": changes_sql,
             }
             self.execute(sql, params)
+        # Add field comment, if required.
+        if (
+            field.db_comment
+            and self.connection.features.supports_comments
+            and not self.connection.features.supports_comments_inline
+        ):
+            field_type = db_params["type"]
+            self.execute(
+                *self._alter_column_comment_sql(
+                    model, field, field_type, field.db_comment
+                )
+            )
         # Add an index, if required
         self.deferred_sql.extend(self._field_indexes_sql(model, field))
         # Reset connection if required
@@ -813,6 +856,11 @@ def _alter_field(
             self.connection.features.supports_foreign_keys
             and old_field.remote_field
             and old_field.db_constraint
+            and self._field_should_be_altered(
+                old_field,
+                new_field,
+                ignore={"db_comment"},
+            )
         ):
             fk_names = self._constraint_names(
                 model, [old_field.column], foreign_key=True
@@ -949,11 +997,15 @@ def _alter_field(
         # Type suffix change? (e.g. auto increment).
         old_type_suffix = old_field.db_type_suffix(connection=self.connection)
         new_type_suffix = new_field.db_type_suffix(connection=self.connection)
-        # Type or collation change?
+        # Type, collation, or comment change?
         if (
             old_type != new_type
             or old_type_suffix != new_type_suffix
             or old_collation != new_collation
+            or (
+                self.connection.features.supports_comments
+                and old_field.db_comment != new_field.db_comment
+            )
         ):
             fragment, other_actions = self._alter_column_type_sql(
                 model, old_field, new_field, new_type, old_collation, new_collation
@@ -1211,12 +1263,26 @@ def _alter_column_type_sql(
         an ALTER TABLE statement and a list of extra (sql, params) tuples to
         run once the field is altered.
         """
+        other_actions = []
         if collate_sql := self._collate_sql(
             new_collation, old_collation, model._meta.db_table
         ):
             collate_sql = f" {collate_sql}"
         else:
             collate_sql = ""
+        # Comment change?
+        comment_sql = ""
+        if self.connection.features.supports_comments and not new_field.many_to_many:
+            if old_field.db_comment != new_field.db_comment:
+                # PostgreSQL and Oracle can't execute 'ALTER COLUMN ...' and
+                # 'COMMENT ON ...' at the same time.
+                sql, params = self._alter_column_comment_sql(
+                    model, new_field, new_type, new_field.db_comment
+                )
+                if sql:
+                    other_actions.append((sql, params))
+            if new_field.db_comment:
+                comment_sql = self._comment_sql(new_field.db_comment)
         return (
             (
                 self.sql_alter_column_type
@@ -1224,12 +1290,27 @@ def _alter_column_type_sql(
                     "column": self.quote_name(new_field.column),
                     "type": new_type,
                     "collation": collate_sql,
+                    "comment": comment_sql,
                 },
                 [],
             ),
+            other_actions,
+        )
+
+    def _alter_column_comment_sql(self, model, new_field, new_type, new_db_comment):
+        return (
+            self.sql_alter_column_comment
+            % {
+                "table": self.quote_name(model._meta.db_table),
+                "column": self.quote_name(new_field.column),
+                "comment": self._comment_sql(new_db_comment),
+            },
             [],
         )
 
+    def _comment_sql(self, comment):
+        return self.quote_value(comment or "")
+
     def _alter_many_to_many(self, model, old_field, new_field, strict):
         """Alter M2Ms to repoint their to= endpoints."""
         # Rename the through table
@@ -1423,16 +1504,18 @@ def _field_indexes_sql(self, model, field):
             output.append(self._create_index_sql(model, fields=[field]))
         return output
 
-    def _field_should_be_altered(self, old_field, new_field):
+    def _field_should_be_altered(self, old_field, new_field, ignore=None):
+        ignore = ignore or set()
         _, old_path, old_args, old_kwargs = old_field.deconstruct()
         _, new_path, new_args, new_kwargs = new_field.deconstruct()
         # Don't alter when:
         # - changing only a field name
         # - changing an attribute that doesn't affect the schema
+        # - changing an attribute in the provided set of ignored attributes
         # - adding only a db_column and the column name is not changed
-        for attr in old_field.non_db_attrs:
+        for attr in ignore.union(old_field.non_db_attrs):
             old_kwargs.pop(attr, None)
-        for attr in new_field.non_db_attrs:
+        for attr in ignore.union(new_field.non_db_attrs):
             new_kwargs.pop(attr, None)
         return self.quote_name(old_field.column) != self.quote_name(
             new_field.column
diff --git a/django/db/backends/mysql/features.py b/django/db/backends/mysql/features.py
--- a/django/db/backends/mysql/features.py
+++ b/django/db/backends/mysql/features.py
@@ -18,6 +18,8 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     requires_explicit_null_ordering_when_grouping = True
     atomic_transactions = False
     can_clone_databases = True
+    supports_comments = True
+    supports_comments_inline = True
     supports_temporal_subtraction = True
     supports_slicing_ordering_in_compound = True
     supports_index_on_text_field = False
diff --git a/django/db/backends/mysql/introspection.py b/django/db/backends/mysql/introspection.py
--- a/django/db/backends/mysql/introspection.py
+++ b/django/db/backends/mysql/introspection.py
@@ -5,18 +5,20 @@
 
 from django.db.backends.base.introspection import BaseDatabaseIntrospection
 from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
-from django.db.backends.base.introspection import TableInfo
+from django.db.backends.base.introspection import TableInfo as BaseTableInfo
 from django.db.models import Index
 from django.utils.datastructures import OrderedSet
 
 FieldInfo = namedtuple(
-    "FieldInfo", BaseFieldInfo._fields + ("extra", "is_unsigned", "has_json_constraint")
+    "FieldInfo",
+    BaseFieldInfo._fields + ("extra", "is_unsigned", "has_json_constraint", "comment"),
 )
 InfoLine = namedtuple(
     "InfoLine",
     "col_name data_type max_len num_prec num_scale extra column_default "
-    "collation is_unsigned",
+    "collation is_unsigned comment",
 )
+TableInfo = namedtuple("TableInfo", BaseTableInfo._fields + ("comment",))
 
 
 class DatabaseIntrospection(BaseDatabaseIntrospection):
@@ -68,9 +70,18 @@ def get_field_type(self, data_type, description):
 
     def get_table_list(self, cursor):
         """Return a list of table and view names in the current database."""
-        cursor.execute("SHOW FULL TABLES")
+        cursor.execute(
+            """
+            SELECT
+                table_name,
+                table_type,
+                table_comment
+            FROM information_schema.tables
+            WHERE table_schema = DATABASE()
+            """
+        )
         return [
-            TableInfo(row[0], {"BASE TABLE": "t", "VIEW": "v"}.get(row[1]))
+            TableInfo(row[0], {"BASE TABLE": "t", "VIEW": "v"}.get(row[1]), row[2])
             for row in cursor.fetchall()
         ]
 
@@ -128,7 +139,8 @@ def get_table_description(self, cursor, table_name):
                 CASE
                     WHEN column_type LIKE '%% unsigned' THEN 1
                     ELSE 0
-                END AS is_unsigned
+                END AS is_unsigned,
+                column_comment
             FROM information_schema.columns
             WHERE table_name = %s AND table_schema = DATABASE()
             """,
@@ -159,6 +171,7 @@ def to_int(i):
                     info.extra,
                     info.is_unsigned,
                     line[0] in json_constraints,
+                    info.comment,
                 )
             )
         return fields
diff --git a/django/db/backends/mysql/schema.py b/django/db/backends/mysql/schema.py
--- a/django/db/backends/mysql/schema.py
+++ b/django/db/backends/mysql/schema.py
@@ -9,7 +9,7 @@ class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):
 
     sql_alter_column_null = "MODIFY %(column)s %(type)s NULL"
     sql_alter_column_not_null = "MODIFY %(column)s %(type)s NOT NULL"
-    sql_alter_column_type = "MODIFY %(column)s %(type)s%(collation)s"
+    sql_alter_column_type = "MODIFY %(column)s %(type)s%(collation)s%(comment)s"
     sql_alter_column_no_default_null = "ALTER COLUMN %(column)s SET DEFAULT NULL"
 
     # No 'CASCADE' which works as a no-op in MySQL but is undocumented
@@ -32,6 +32,9 @@ class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):
 
     sql_create_index = "CREATE INDEX %(name)s ON %(table)s (%(columns)s)%(extra)s"
 
+    sql_alter_table_comment = "ALTER TABLE %(table)s COMMENT = %(comment)s"
+    sql_alter_column_comment = None
+
     @property
     def sql_delete_check(self):
         if self.connection.mysql_is_mariadb:
@@ -228,3 +231,11 @@ def _alter_column_type_sql(
     def _rename_field_sql(self, table, old_field, new_field, new_type):
         new_type = self._set_field_new_type_null_status(old_field, new_type)
         return super()._rename_field_sql(table, old_field, new_field, new_type)
+
+    def _alter_column_comment_sql(self, model, new_field, new_type, new_db_comment):
+        # Comment is alter when altering the column type.
+        return "", []
+
+    def _comment_sql(self, comment):
+        comment_sql = super()._comment_sql(comment)
+        return f" COMMENT {comment_sql}"
diff --git a/django/db/backends/oracle/features.py b/django/db/backends/oracle/features.py
--- a/django/db/backends/oracle/features.py
+++ b/django/db/backends/oracle/features.py
@@ -25,6 +25,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     supports_partially_nullable_unique_constraints = False
     supports_deferrable_unique_constraints = True
     truncates_names = True
+    supports_comments = True
     supports_tablespaces = True
     supports_sequence_reset = False
     can_introspect_materialized_views = True
diff --git a/django/db/backends/oracle/introspection.py b/django/db/backends/oracle/introspection.py
--- a/django/db/backends/oracle/introspection.py
+++ b/django/db/backends/oracle/introspection.py
@@ -5,10 +5,13 @@
 from django.db import models
 from django.db.backends.base.introspection import BaseDatabaseIntrospection
 from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
-from django.db.backends.base.introspection import TableInfo
+from django.db.backends.base.introspection import TableInfo as BaseTableInfo
 from django.utils.functional import cached_property
 
-FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield", "is_json"))
+FieldInfo = namedtuple(
+    "FieldInfo", BaseFieldInfo._fields + ("is_autofield", "is_json", "comment")
+)
+TableInfo = namedtuple("TableInfo", BaseTableInfo._fields + ("comment",))
 
 
 class DatabaseIntrospection(BaseDatabaseIntrospection):
@@ -77,8 +80,14 @@ def get_table_list(self, cursor):
         """Return a list of table and view names in the current database."""
         cursor.execute(
             """
-            SELECT table_name, 't'
+            SELECT
+                user_tables.table_name,
+                't',
+                user_tab_comments.comments
             FROM user_tables
+            LEFT OUTER JOIN
+                user_tab_comments
+                ON user_tab_comments.table_name = user_tables.table_name
             WHERE
                 NOT EXISTS (
                     SELECT 1
@@ -86,13 +95,13 @@ def get_table_list(self, cursor):
                     WHERE user_mviews.mview_name = user_tables.table_name
                 )
             UNION ALL
-            SELECT view_name, 'v' FROM user_views
+            SELECT view_name, 'v', NULL FROM user_views
             UNION ALL
-            SELECT mview_name, 'v' FROM user_mviews
+            SELECT mview_name, 'v', NULL FROM user_mviews
         """
         )
         return [
-            TableInfo(self.identifier_converter(row[0]), row[1])
+            TableInfo(self.identifier_converter(row[0]), row[1], row[2])
             for row in cursor.fetchall()
         ]
 
@@ -131,10 +140,15 @@ def get_table_description(self, cursor, table_name):
                     )
                     THEN 1
                     ELSE 0
-                END as is_json
+                END as is_json,
+                user_col_comments.comments as col_comment
             FROM user_tab_cols
             LEFT OUTER JOIN
                 user_tables ON user_tables.table_name = user_tab_cols.table_name
+            LEFT OUTER JOIN
+                user_col_comments ON
+                user_col_comments.column_name = user_tab_cols.column_name AND
+                user_col_comments.table_name = user_tab_cols.table_name
             WHERE user_tab_cols.table_name = UPPER(%s)
             """,
             [table_name],
@@ -146,6 +160,7 @@ def get_table_description(self, cursor, table_name):
                 collation,
                 is_autofield,
                 is_json,
+                comment,
             )
             for (
                 column,
@@ -154,6 +169,7 @@ def get_table_description(self, cursor, table_name):
                 display_size,
                 is_autofield,
                 is_json,
+                comment,
             ) in cursor.fetchall()
         }
         self.cache_bust_counter += 1
@@ -165,7 +181,14 @@ def get_table_description(self, cursor, table_name):
         description = []
         for desc in cursor.description:
             name = desc[0]
-            display_size, default, collation, is_autofield, is_json = field_map[name]
+            (
+                display_size,
+                default,
+                collation,
+                is_autofield,
+                is_json,
+                comment,
+            ) = field_map[name]
             name %= {}  # cx_Oracle, for some reason, doubles percent signs.
             description.append(
                 FieldInfo(
@@ -180,6 +203,7 @@ def get_table_description(self, cursor, table_name):
                     collation,
                     is_autofield,
                     is_json,
+                    comment,
                 )
             )
         return description
diff --git a/django/db/backends/postgresql/features.py b/django/db/backends/postgresql/features.py
--- a/django/db/backends/postgresql/features.py
+++ b/django/db/backends/postgresql/features.py
@@ -22,6 +22,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     has_select_for_update_skip_locked = True
     has_select_for_no_key_update = True
     can_release_savepoints = True
+    supports_comments = True
     supports_tablespaces = True
     supports_transactions = True
     can_introspect_materialized_views = True
diff --git a/django/db/backends/postgresql/introspection.py b/django/db/backends/postgresql/introspection.py
--- a/django/db/backends/postgresql/introspection.py
+++ b/django/db/backends/postgresql/introspection.py
@@ -2,10 +2,11 @@
 
 from django.db.backends.base.introspection import BaseDatabaseIntrospection
 from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
-from django.db.backends.base.introspection import TableInfo
+from django.db.backends.base.introspection import TableInfo as BaseTableInfo
 from django.db.models import Index
 
-FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield",))
+FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield", "comment"))
+TableInfo = namedtuple("TableInfo", BaseTableInfo._fields + ("comment",))
 
 
 class DatabaseIntrospection(BaseDatabaseIntrospection):
@@ -62,7 +63,8 @@ def get_table_list(self, cursor):
                     WHEN c.relispartition THEN 'p'
                     WHEN c.relkind IN ('m', 'v') THEN 'v'
                     ELSE 't'
-                END
+                END,
+                obj_description(c.oid)
             FROM pg_catalog.pg_class c
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
             WHERE c.relkind IN ('f', 'm', 'p', 'r', 'v')
@@ -91,7 +93,8 @@ def get_table_description(self, cursor, table_name):
                 NOT (a.attnotnull OR (t.typtype = 'd' AND t.typnotnull)) AS is_nullable,
                 pg_get_expr(ad.adbin, ad.adrelid) AS column_default,
                 CASE WHEN collname = 'default' THEN NULL ELSE collname END AS collation,
-                a.attidentity != '' AS is_autofield
+                a.attidentity != '' AS is_autofield,
+                col_description(a.attrelid, a.attnum) AS column_comment
             FROM pg_attribute a
             LEFT JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
             LEFT JOIN pg_collation co ON a.attcollation = co.oid
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -170,6 +170,7 @@ def _detect_changes(self, convert_apps=None, graph=None):
         self.generate_created_proxies()
         self.generate_altered_options()
         self.generate_altered_managers()
+        self.generate_altered_db_table_comment()
 
         # Create the renamed fields and store them in self.renamed_fields.
         # They are used by create_altered_indexes(), generate_altered_fields(),
@@ -1552,6 +1553,28 @@ def generate_altered_db_table(self):
                     ),
                 )
 
+    def generate_altered_db_table_comment(self):
+        models_to_check = self.kept_model_keys.union(
+            self.kept_proxy_keys, self.kept_unmanaged_keys
+        )
+        for app_label, model_name in sorted(models_to_check):
+            old_model_name = self.renamed_models.get(
+                (app_label, model_name), model_name
+            )
+            old_model_state = self.from_state.models[app_label, old_model_name]
+            new_model_state = self.to_state.models[app_label, model_name]
+
+            old_db_table_comment = old_model_state.options.get("db_table_comment")
+            new_db_table_comment = new_model_state.options.get("db_table_comment")
+            if old_db_table_comment != new_db_table_comment:
+                self.add_operation(
+                    app_label,
+                    operations.AlterModelTableComment(
+                        name=model_name,
+                        table_comment=new_db_table_comment,
+                    ),
+                )
+
     def generate_altered_options(self):
         """
         Work out if any non-schema-affecting options have changed and make an
diff --git a/django/db/migrations/operations/__init__.py b/django/db/migrations/operations/__init__.py
--- a/django/db/migrations/operations/__init__.py
+++ b/django/db/migrations/operations/__init__.py
@@ -6,6 +6,7 @@
     AlterModelManagers,
     AlterModelOptions,
     AlterModelTable,
+    AlterModelTableComment,
     AlterOrderWithRespectTo,
     AlterUniqueTogether,
     CreateModel,
@@ -21,6 +22,7 @@
     "CreateModel",
     "DeleteModel",
     "AlterModelTable",
+    "AlterModelTableComment",
     "AlterUniqueTogether",
     "RenameModel",
     "AlterIndexTogether",
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -529,6 +529,44 @@ def migration_name_fragment(self):
         return "alter_%s_table" % self.name_lower
 
 
+class AlterModelTableComment(ModelOptionOperation):
+    def __init__(self, name, table_comment):
+        self.table_comment = table_comment
+        super().__init__(name)
+
+    def deconstruct(self):
+        kwargs = {
+            "name": self.name,
+            "table_comment": self.table_comment,
+        }
+        return (self.__class__.__qualname__, [], kwargs)
+
+    def state_forwards(self, app_label, state):
+        state.alter_model_options(
+            app_label, self.name_lower, {"db_table_comment": self.table_comment}
+        )
+
+    def database_forwards(self, app_label, schema_editor, from_state, to_state):
+        new_model = to_state.apps.get_model(app_label, self.name)
+        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
+            old_model = from_state.apps.get_model(app_label, self.name)
+            schema_editor.alter_db_table_comment(
+                new_model,
+                old_model._meta.db_table_comment,
+                new_model._meta.db_table_comment,
+            )
+
+    def database_backwards(self, app_label, schema_editor, from_state, to_state):
+        return self.database_forwards(app_label, schema_editor, from_state, to_state)
+
+    def describe(self):
+        return f"Alter {self.name} table comment"
+
+    @property
+    def migration_name_fragment(self):
+        return f"alter_{self.name_lower}_table_comment"
+
+
 class AlterTogetherOptionOperation(ModelOptionOperation):
     option_name = None
 
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -1556,6 +1556,7 @@ def check(cls, **kwargs):
                 *cls._check_ordering(),
                 *cls._check_constraints(databases),
                 *cls._check_default_pk(),
+                *cls._check_db_table_comment(databases),
             ]
 
         return errors
@@ -1592,6 +1593,29 @@ def _check_default_pk(cls):
             ]
         return []
 
+    @classmethod
+    def _check_db_table_comment(cls, databases):
+        if not cls._meta.db_table_comment:
+            return []
+        errors = []
+        for db in databases:
+            if not router.allow_migrate_model(db, cls):
+                continue
+            connection = connections[db]
+            if not (
+                connection.features.supports_comments
+                or "supports_comments" in cls._meta.required_db_features
+            ):
+                errors.append(
+                    checks.Warning(
+                        f"{connection.display_name} does not support comments on "
+                        f"tables (db_table_comment).",
+                        obj=cls,
+                        id="models.W046",
+                    )
+                )
+        return errors
+
     @classmethod
     def _check_swappable(cls):
         """Check if the swapped model exists."""
diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -200,6 +200,7 @@ def __init__(
         auto_created=False,
         validators=(),
         error_messages=None,
+        db_comment=None,
     ):
         self.name = name
         self.verbose_name = verbose_name  # May be set by set_attributes_from_name
@@ -221,6 +222,7 @@ def __init__(
         self.help_text = help_text
         self.db_index = db_index
         self.db_column = db_column
+        self.db_comment = db_comment
         self._db_tablespace = db_tablespace
         self.auto_created = auto_created
 
@@ -259,6 +261,7 @@ def check(self, **kwargs):
             *self._check_field_name(),
             *self._check_choices(),
             *self._check_db_index(),
+            *self._check_db_comment(**kwargs),
             *self._check_null_allowed_for_primary_keys(),
             *self._check_backend_specific_checks(**kwargs),
             *self._check_validators(),
@@ -385,6 +388,28 @@ def _check_db_index(self):
         else:
             return []
 
+    def _check_db_comment(self, databases=None, **kwargs):
+        if not self.db_comment or not databases:
+            return []
+        errors = []
+        for db in databases:
+            if not router.allow_migrate_model(db, self.model):
+                continue
+            connection = connections[db]
+            if not (
+                connection.features.supports_comments
+                or "supports_comments" in self.model._meta.required_db_features
+            ):
+                errors.append(
+                    checks.Warning(
+                        f"{connection.display_name} does not support comments on "
+                        f"columns (db_comment).",
+                        obj=self,
+                        id="fields.W163",
+                    )
+                )
+        return errors
+
     def _check_null_allowed_for_primary_keys(self):
         if (
             self.primary_key
@@ -538,6 +563,7 @@ def deconstruct(self):
             "choices": None,
             "help_text": "",
             "db_column": None,
+            "db_comment": None,
             "db_tablespace": None,
             "auto_created": False,
             "validators": [],
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -1428,6 +1428,14 @@ def _check_ignored_options(self, **kwargs):
                     id="fields.W345",
                 )
             )
+        if self.db_comment:
+            warnings.append(
+                checks.Warning(
+                    "db_comment has no effect on ManyToManyField.",
+                    obj=self,
+                    id="fields.W346",
+                )
+            )
 
         return warnings
 
diff --git a/django/db/models/options.py b/django/db/models/options.py
--- a/django/db/models/options.py
+++ b/django/db/models/options.py
@@ -30,6 +30,7 @@
     "verbose_name",
     "verbose_name_plural",
     "db_table",
+    "db_table_comment",
     "ordering",
     "unique_together",
     "permissions",
@@ -112,6 +113,7 @@ def __init__(self, meta, app_label=None):
         self.verbose_name = None
         self.verbose_name_plural = None
         self.db_table = ""
+        self.db_table_comment = ""
         self.ordering = []
         self._ordering_clash = False
         self.indexes = []

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/commands/inspectdb.py | 81 | 92 | 47 | 12 | 17772
| django/core/management/commands/inspectdb.py | 235 | 235 | 47 | 12 | 17772
| django/core/management/commands/inspectdb.py | 245 | 252 | 47 | 12 | 17772
| django/core/management/commands/inspectdb.py | 356 | 356 | 114 | 12 | 41310
| django/core/management/commands/inspectdb.py | 394 | 394 | 114 | 12 | 41310
| django/db/backends/base/features.py | 337 | 337 | 144 | 8 | 50515
| django/db/backends/base/schema.py | 144 | 144 | 8 | 5 | 3748
| django/db/backends/base/schema.py | 292 | 292 | 31 | 5 | 12008
| django/db/backends/base/schema.py | 448 | 448 | 44 | 5 | 15773
| django/db/backends/base/schema.py | 617 | 617 | 36 | 5 | 13545
| django/db/backends/base/schema.py | 696 | 696 | 38 | 5 | 14386
| django/db/backends/base/schema.py | 816 | 816 | - | 5 | -
| django/db/backends/base/schema.py | 952 | 952 | 18 | 5 | 7106
| django/db/backends/base/schema.py | 1214 | 1214 | 66 | 5 | 25511
| django/db/backends/base/schema.py | 1227 | 1227 | 66 | 5 | 25511
| django/db/backends/base/schema.py | 1426 | 1435 | - | 5 | -
| django/db/backends/mysql/features.py | 21 | 21 | 55 | 6 | 20543
| django/db/backends/mysql/introspection.py | 8 | 18 | 121 | 11 | 42977
| django/db/backends/mysql/introspection.py | 71 | 73 | - | 11 | -
| django/db/backends/mysql/introspection.py | 131 | 131 | 60 | 11 | 23216
| django/db/backends/mysql/introspection.py | 162 | 162 | 60 | 11 | 23216
| django/db/backends/mysql/schema.py | 12 | 12 | 1 | 1 | 453
| django/db/backends/mysql/schema.py | 35 | 35 | 1 | 1 | 453
| django/db/backends/mysql/schema.py | 231 | 231 | 74 | 1 | 28061
| django/db/backends/oracle/features.py | 28 | 28 | 97 | 28 | 35512
| django/db/backends/oracle/introspection.py | 8 | 11 | 137 | 19 | 49077
| django/db/backends/oracle/introspection.py | 80 | 80 | - | 19 | -
| django/db/backends/oracle/introspection.py | 89 | 95 | - | 19 | -
| django/db/backends/oracle/introspection.py | 134 | 134 | 65 | 19 | 25280
| django/db/backends/oracle/introspection.py | 149 | 149 | 65 | 19 | 25280
| django/db/backends/oracle/introspection.py | 157 | 157 | 65 | 19 | 25280
| django/db/backends/oracle/introspection.py | 168 | 168 | 65 | 19 | 25280
| django/db/backends/oracle/introspection.py | 183 | 183 | 65 | 19 | 25280
| django/db/backends/postgresql/features.py | 25 | 25 | 56 | 16 | 21430
| django/db/backends/postgresql/introspection.py | 5 | 8 | 94 | 17 | 34112
| django/db/backends/postgresql/introspection.py | 65 | 65 | - | 17 | -
| django/db/backends/postgresql/introspection.py | 94 | 94 | 59 | 17 | 22570
| django/db/migrations/autodetector.py | 173 | 173 | - | 38 | -
| django/db/migrations/autodetector.py | 1555 | 1555 | - | 38 | -
| django/db/migrations/operations/__init__.py | 9 | 9 | 108 | 32 | 39177
| django/db/migrations/operations/__init__.py | 24 | 24 | 108 | 32 | 39177
| django/db/migrations/operations/models.py | 532 | 532 | - | 24 | -
| django/db/models/base.py | 1559 | 1559 | 72 | 10 | 27586
| django/db/models/base.py | 1595 | 1595 | - | 10 | -
| django/db/models/fields/__init__.py | 203 | 203 | - | 31 | -
| django/db/models/fields/__init__.py | 224 | 224 | - | 31 | -
| django/db/models/fields/__init__.py | 262 | 262 | - | 31 | -
| django/db/models/fields/__init__.py | 388 | 388 | - | 31 | -
| django/db/models/fields/__init__.py | 541 | 541 | - | 31 | -
| django/db/models/fields/related.py | 1431 | 1431 | - | 42 | -
| django/db/models/options.py | 33 | 33 | 101 | 15 | 36834
| django/db/models/options.py | 115 | 115 | - | 15 | -


## Problem Statement

```
Add the ability to define comments in table / columns
Description
	 
		(last modified by Jared Chung)
	 
Database-level comments are valuable for database administrators, data analysts, data scientists, and others who are looking to consume data that is managed by Django. Most Django-supported databases also support table-level and column-level comments. This ticket would add functionality to Django to allow Django users to specify comments for syncdb manage.py to enter into the database.
....
....
new proposal (kimsoungryoul : 2020.03.23)
We will develop the code such as below
class AModel(models.Model):
	 aaa = model.CharField(help_text="i am help_text", db_column_comment="i am db_comment",~~~)
	 
	 class Meta:
		 db_table = "a_model_example_name"
		 db_table_comment ="this is a_model comment ~~~~"

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/backends/mysql/schema.py** | 1 | 42| 453 | 453 | 1960 | 
| 2 | 2 django/db/backends/sqlite3/schema.py | 100 | 121| 195 | 648 | 6675 | 
| 3 | **2 django/db/backends/mysql/schema.py** | 104 | 118| 144 | 792 | 6675 | 
| 4 | 3 django/db/backends/postgresql/schema.py | 143 | 256| 920 | 1712 | 9510 | 
| 5 | 4 django/db/backends/oracle/schema.py | 1 | 28| 271 | 1983 | 11840 | 
| 6 | 4 django/db/backends/postgresql/schema.py | 1 | 84| 751 | 2734 | 11840 | 
| 7 | 4 django/db/backends/postgresql/schema.py | 313 | 338| 235 | 2969 | 11840 | 
| **-> 8 <-** | **5 django/db/backends/base/schema.py** | 75 | 151| 779 | 3748 | 25601 | 
| 9 | 5 django/db/backends/postgresql/schema.py | 277 | 311| 277 | 4025 | 25601 | 
| 10 | 5 django/db/backends/postgresql/schema.py | 258 | 275| 170 | 4195 | 25601 | 
| 11 | **5 django/db/backends/mysql/schema.py** | 44 | 53| 134 | 4329 | 25601 | 
| 12 | **5 django/db/backends/mysql/schema.py** | 55 | 102| 398 | 4727 | 25601 | 
| 13 | 5 django/db/backends/sqlite3/schema.py | 25 | 41| 168 | 4895 | 25601 | 
| 14 | **5 django/db/backends/base/schema.py** | 1639 | 1663| 189 | 5084 | 25601 | 
| 15 | 5 django/db/backends/sqlite3/schema.py | 553 | 577| 162 | 5246 | 25601 | 
| 16 | 5 django/db/backends/sqlite3/schema.py | 256 | 361| 885 | 6131 | 25601 | 
| 17 | 5 django/db/backends/postgresql/schema.py | 122 | 141| 234 | 6365 | 25601 | 
| **-> 18 <-** | **5 django/db/backends/base/schema.py** | 889 | 970| 741 | 7106 | 25601 | 
| 19 | 5 django/db/backends/postgresql/schema.py | 340 | 376| 210 | 7316 | 25601 | 
| 20 | 5 django/db/backends/oracle/schema.py | 171 | 184| 166 | 7482 | 25601 | 
| 21 | **5 django/db/backends/base/schema.py** | 971 | 1058| 822 | 8304 | 25601 | 
| 22 | **6 django/db/backends/mysql/features.py** | 147 | 254| 728 | 9032 | 27922 | 
| 23 | **6 django/db/backends/base/schema.py** | 1547 | 1598| 343 | 9375 | 27922 | 
| 24 | **6 django/db/backends/base/schema.py** | 1059 | 1133| 725 | 10100 | 27922 | 
| 25 | 6 django/db/backends/sqlite3/schema.py | 363 | 379| 132 | 10232 | 27922 | 
| 26 | 6 django/db/backends/sqlite3/schema.py | 123 | 174| 527 | 10759 | 27922 | 
| 27 | 6 django/db/backends/oracle/schema.py | 229 | 252| 194 | 10953 | 27922 | 
| 28 | 7 django/db/backends/sqlite3/features.py | 132 | 166| 245 | 11198 | 29266 | 
| 29 | 7 django/db/backends/sqlite3/schema.py | 381 | 400| 176 | 11374 | 29266 | 
| 30 | **8 django/db/backends/base/features.py** | 368 | 386| 173 | 11547 | 32423 | 
| **-> 31 <-** | **8 django/db/backends/base/schema.py** | 284 | 339| 461 | 12008 | 32423 | 
| 32 | 8 django/db/backends/sqlite3/features.py | 1 | 61| 584 | 12592 | 32423 | 
| 33 | **8 django/db/backends/base/schema.py** | 1376 | 1392| 153 | 12745 | 32423 | 
| 34 | **8 django/db/backends/base/schema.py** | 153 | 201| 380 | 13125 | 32423 | 
| 35 | **8 django/db/backends/base/schema.py** | 1441 | 1460| 175 | 13300 | 32423 | 
| **-> 36 <-** | **8 django/db/backends/base/schema.py** | 598 | 626| 245 | 13545 | 32423 | 
| 37 | **8 django/db/backends/base/schema.py** | 1462 | 1481| 176 | 13721 | 32423 | 
| **-> 38 <-** | **8 django/db/backends/base/schema.py** | 628 | 700| 665 | 14386 | 32423 | 
| 39 | 8 django/db/backends/oracle/schema.py | 214 | 227| 115 | 14501 | 32423 | 
| 40 | 8 django/db/backends/sqlite3/schema.py | 1 | 23| 191 | 14692 | 32423 | 
| 41 | 9 django/db/backends/base/creation.py | 353 | 381| 216 | 14908 | 35364 | 
| 42 | 9 django/db/backends/oracle/schema.py | 74 | 102| 344 | 15252 | 35364 | 
| 43 | **10 django/db/models/base.py** | 1 | 66| 361 | 15613 | 53938 | 
| **-> 44 <-** | **10 django/db/backends/base/schema.py** | 438 | 455| 160 | 15773 | 53938 | 
| 45 | **10 django/db/backends/base/schema.py** | 478 | 520| 316 | 16089 | 53938 | 
| 46 | **11 django/db/backends/mysql/introspection.py** | 214 | 225| 121 | 16210 | 56528 | 
| **-> 47 <-** | **12 django/core/management/commands/inspectdb.py** | 54 | 253| 1562 | 17772 | 59500 | 
| 48 | 13 django/contrib/gis/db/backends/postgis/schema.py | 1 | 23| 217 | 17989 | 60198 | 
| 49 | **13 django/db/backends/base/schema.py** | 1600 | 1637| 243 | 18232 | 60198 | 
| 50 | **13 django/db/backends/base/schema.py** | 1483 | 1505| 199 | 18431 | 60198 | 
| 51 | **13 django/db/backends/mysql/features.py** | 256 | 324| 504 | 18935 | 60198 | 
| 52 | 13 django/db/backends/oracle/schema.py | 30 | 49| 206 | 19141 | 60198 | 
| 53 | 14 django/contrib/gis/db/backends/spatialite/schema.py | 1 | 37| 320 | 19461 | 61592 | 
| 54 | **15 django/db/models/options.py** | 173 | 242| 645 | 20106 | 69275 | 
| **-> 55 <-** | **15 django/db/backends/mysql/features.py** | 1 | 61| 437 | 20543 | 69275 | 
| **-> 56 <-** | **16 django/db/backends/postgresql/features.py** | 1 | 112| 887 | 21430 | 70162 | 
| 57 | 16 django/db/backends/oracle/schema.py | 51 | 72| 146 | 21576 | 70162 | 
| 58 | **16 django/db/backends/base/schema.py** | 734 | 796| 524 | 22100 | 70162 | 
| **-> 59 <-** | **17 django/db/backends/postgresql/introspection.py** | 79 | 124| 470 | 22570 | 72614 | 
| **-> 60 <-** | **17 django/db/backends/mysql/introspection.py** | 77 | 164| 646 | 23216 | 72614 | 
| 61 | **17 django/db/backends/mysql/features.py** | 83 | 145| 500 | 23716 | 72614 | 
| 62 | 17 django/db/backends/postgresql/schema.py | 86 | 120| 332 | 24048 | 72614 | 
| 63 | **17 django/db/models/base.py** | 369 | 436| 541 | 24589 | 72614 | 
| 64 | 18 django/db/models/sql/query.py | 1076 | 1091| 135 | 24724 | 95363 | 
| **-> 65 <-** | **19 django/db/backends/oracle/introspection.py** | 99 | 185| 556 | 25280 | 98099 | 
| **-> 66 <-** | **19 django/db/backends/base/schema.py** | 1202 | 1231| 231 | 25511 | 98099 | 
| 67 | **19 django/db/backends/base/schema.py** | 1706 | 1743| 317 | 25828 | 98099 | 
| 68 | **19 django/db/backends/base/schema.py** | 1507 | 1545| 240 | 26068 | 98099 | 
| 69 | 19 django/db/backends/sqlite3/schema.py | 43 | 72| 245 | 26313 | 98099 | 
| 70 | **19 django/db/models/base.py** | 2160 | 2241| 592 | 26905 | 98099 | 
| 71 | 19 django/contrib/gis/db/backends/spatialite/schema.py | 137 | 192| 404 | 27309 | 98099 | 
| **-> 72 <-** | **19 django/db/models/base.py** | 1526 | 1561| 277 | 27586 | 98099 | 
| 73 | **19 django/db/backends/base/schema.py** | 1297 | 1320| 215 | 27801 | 98099 | 
| **-> 74 <-** | **19 django/db/backends/mysql/schema.py** | 205 | 231| 260 | 28061 | 98099 | 
| 75 | 20 django/contrib/gis/db/backends/mysql/schema.py | 33 | 55| 197 | 28258 | 98756 | 
| 76 | 21 django/db/backends/sqlite3/introspection.py | 265 | 302| 294 | 28552 | 102081 | 
| 77 | **21 django/db/backends/mysql/schema.py** | 137 | 153| 144 | 28696 | 102081 | 
| 78 | **21 django/db/backends/base/schema.py** | 1164 | 1200| 269 | 28965 | 102081 | 
| 79 | 21 django/contrib/gis/db/backends/mysql/schema.py | 1 | 31| 233 | 29198 | 102081 | 
| 80 | **21 django/db/backends/base/schema.py** | 203 | 282| 651 | 29849 | 102081 | 
| 81 | 22 django/contrib/admin/models.py | 24 | 45| 122 | 29971 | 103274 | 
| 82 | **22 django/db/models/base.py** | 938 | 1026| 690 | 30661 | 103274 | 
| 83 | 23 django/contrib/admin/options.py | 2466 | 2501| 315 | 30976 | 122513 | 
| 84 | **24 django/db/migrations/operations/models.py** | 689 | 741| 320 | 31296 | 130226 | 
| 85 | 24 django/db/backends/sqlite3/introspection.py | 88 | 133| 315 | 31611 | 130226 | 
| 86 | 25 django/db/backends/ddl_references.py | 44 | 76| 208 | 31819 | 131855 | 
| 87 | **25 django/db/backends/mysql/schema.py** | 120 | 135| 116 | 31935 | 131855 | 
| 88 | 26 django/db/migrations/questioner.py | 249 | 267| 177 | 32112 | 134551 | 
| 89 | **26 django/db/backends/base/schema.py** | 564 | 596| 268 | 32380 | 134551 | 
| 90 | 26 django/db/backends/sqlite3/introspection.py | 165 | 263| 767 | 33147 | 134551 | 
| 91 | 26 django/contrib/admin/models.py | 48 | 85| 250 | 33397 | 134551 | 
| 92 | 26 django/db/backends/ddl_references.py | 115 | 135| 167 | 33564 | 134551 | 
| 93 | 27 django/db/backends/mysql/compiler.py | 55 | 85| 240 | 33804 | 135198 | 
| **-> 94 <-** | **27 django/db/backends/postgresql/introspection.py** | 1 | 38| 308 | 34112 | 135198 | 
| 95 | **27 django/db/backends/base/schema.py** | 398 | 420| 176 | 34288 | 135198 | 
| 96 | **27 django/core/management/commands/inspectdb.py** | 255 | 313| 487 | 34775 | 135198 | 
| **-> 97 <-** | **28 django/db/backends/oracle/features.py** | 1 | 86| 737 | 35512 | 136534 | 
| 98 | 29 django/core/management/commands/createcachetable.py | 42 | 54| 121 | 35633 | 137425 | 
| 99 | 29 django/db/backends/sqlite3/features.py | 63 | 130| 528 | 36161 | 137425 | 
| 100 | **29 django/db/models/base.py** | 574 | 612| 326 | 36487 | 137425 | 
| **-> 101 <-** | **29 django/db/models/options.py** | 1 | 57| 347 | 36834 | 137425 | 
| 102 | 29 django/db/backends/ddl_references.py | 223 | 255| 252 | 37086 | 137425 | 
| 103 | 30 django/contrib/gis/db/backends/oracle/schema.py | 74 | 122| 323 | 37409 | 138336 | 
| 104 | **31 django/db/models/fields/__init__.py** | 2332 | 2366| 229 | 37638 | 157050 | 
| 105 | **31 django/db/models/base.py** | 1350 | 1379| 294 | 37932 | 157050 | 
| 106 | 31 django/contrib/admin/options.py | 1748 | 1850| 780 | 38712 | 157050 | 
| 107 | **31 django/db/models/base.py** | 1123 | 1150| 238 | 38950 | 157050 | 
| **-> 108 <-** | **32 django/db/migrations/operations/__init__.py** | 1 | 43| 227 | 39177 | 157277 | 
| 109 | **32 django/db/models/base.py** | 459 | 572| 957 | 40134 | 157277 | 
| 110 | 33 django/db/backends/postgresql/creation.py | 1 | 38| 271 | 40405 | 157966 | 
| 111 | 34 django/db/backends/mysql/creation.py | 1 | 29| 221 | 40626 | 158635 | 
| 112 | 34 django/contrib/admin/models.py | 1 | 21| 123 | 40749 | 158635 | 
| 113 | 34 django/contrib/gis/db/backends/oracle/schema.py | 44 | 72| 237 | 40986 | 158635 | 
| **-> 114 <-** | **34 django/core/management/commands/inspectdb.py** | 355 | 395| 324 | 41310 | 158635 | 
| 115 | 34 django/core/management/commands/createcachetable.py | 56 | 131| 552 | 41862 | 158635 | 
| 116 | 34 django/contrib/admin/options.py | 948 | 974| 212 | 42074 | 158635 | 
| 117 | 34 django/db/migrations/questioner.py | 166 | 187| 188 | 42262 | 158635 | 
| 118 | 35 django/db/backends/postgresql/operations.py | 25 | 56| 194 | 42456 | 162020 | 
| 119 | **35 django/db/backends/mysql/features.py** | 63 | 81| 177 | 42633 | 162020 | 
| 120 | **35 django/db/backends/base/schema.py** | 368 | 396| 205 | 42838 | 162020 | 
| **-> 121 <-** | **35 django/db/backends/mysql/introspection.py** | 1 | 19| 139 | 42977 | 162020 | 
| 122 | 36 django/db/backends/mysql/validation.py | 1 | 36| 246 | 43223 | 162551 | 
| **-> 123 <-** | **36 django/db/backends/base/features.py** | 223 | 366| 1259 | 44482 | 162551 | 
| 124 | 36 django/db/backends/sqlite3/schema.py | 176 | 255| 760 | 45242 | 162551 | 
| 125 | **36 django/db/backends/mysql/introspection.py** | 22 | 45| 227 | 45469 | 162551 | 
| 126 | **36 django/db/migrations/operations/models.py** | 41 | 111| 524 | 45993 | 162551 | 
| 127 | **36 django/db/backends/base/schema.py** | 1135 | 1162| 195 | 46188 | 162551 | 
| 128 | 37 django/db/models/__init__.py | 1 | 116| 682 | 46870 | 163233 | 
| 129 | **37 django/db/backends/postgresql/introspection.py** | 265 | 297| 223 | 47093 | 163233 | 
| 130 | 37 django/contrib/gis/db/backends/spatialite/schema.py | 39 | 66| 210 | 47303 | 163233 | 
| 131 | **37 django/db/models/options.py** | 441 | 463| 164 | 47467 | 163233 | 
| 132 | **38 django/db/migrations/autodetector.py** | 1534 | 1553| 187 | 47654 | 176694 | 
| 133 | 38 django/contrib/admin/options.py | 117 | 147| 223 | 47877 | 176694 | 
| 134 | **38 django/db/models/base.py** | 653 | 673| 176 | 48053 | 176694 | 
| 135 | 39 django/db/backends/base/operations.py | 752 | 778| 222 | 48275 | 182724 | 
| 136 | **39 django/db/models/options.py** | 287 | 329| 355 | 48630 | 182724 | 
| **-> 137 <-** | **39 django/db/backends/oracle/introspection.py** | 1 | 49| 447 | 49077 | 182724 | 
| 138 | **39 django/db/models/base.py** | 1998 | 2051| 359 | 49436 | 182724 | 
| 139 | 40 django/utils/translation/__init__.py | 1 | 43| 255 | 49691 | 184600 | 
| 140 | **40 django/db/backends/mysql/introspection.py** | 313 | 337| 247 | 49938 | 184600 | 
| 141 | 41 django/db/models/sql/__init__.py | 1 | 7| 0 | 49938 | 184666 | 
| 142 | **41 django/db/models/base.py** | 1196 | 1236| 310 | 50248 | 184666 | 
| 143 | **41 django/db/models/base.py** | 1563 | 1593| 247 | 50495 | 184666 | 
| **-> 144 <-** | **41 django/db/backends/base/features.py** | 1 | 385| 20 | 50515 | 184666 | 
| 145 | **42 django/db/models/fields/related.py** | 901 | 990| 583 | 51098 | 199170 | 
| 146 | **42 django/db/models/options.py** | 331 | 367| 338 | 51436 | 199170 | 


### Hint

```
Initial implementation for postgresql_psycopg2 backend: ​https://github.com/niwibe/django/compare/issue_18468 Is only implemented the table comment sql. If accepted as a new feature, would be nice to make design decisions. Today I put the comment for table in class "Meta", another option is to use the first line of the model docstrings. For fields, is more simple, just add an optional parameter "comment" at the Field.
This is a duplicate of #13867, which lingered in DDN for two years until I wontfix'd it. The only argument I've seen is "it'd be nice if...". But I'm against adding features to Django just because we can; there must be a use case. Could you start a thread on the mailing-list, as recommended in the ​contributing guide, to see if there is support for this idea? If there's a good explanation of why you need this, I can reconsider my decision.
I understand your opinion! In any case, I'll write an email to the list which I think is useful. Thanks!
I assume that most databases have comment on fields / tables feature not just because it is nice. Accessing Django created tables from another tool (for instance pgAdmin) would just make the system more productive. Personally I never create SQL tables without proper comments in the database itself.
The reasons given on #13867 still stand — Django is not aiming to provide a wrapper for every SQL feature. For example, it also doesn't provide an easy way to create stored procedures, functions or views, but you can always execute the SQL manually to add these, or could add some additional Python code that executed the SQL — for example using a South migration — if you want to ensure it always happens. In addition, if the audience of these comments is people administering the database without reading the Python source code, it doesn't make sense for these comments to be taking up space in the Python code, which has its own way of adding comments (docstrings and comment lines), which are targeted at programmers not DB admins.
Now that migration is built in Django, which can be used to create and administer the database, I again request this feature to be added. Thanks.
The correct way to reopen a ticket closed as "wontfix" is to start a discussion on the DevelopersMailingList. If there is consensus there to add the feature, then we reopen the ticket.
After discussion on mailing list, the feature is good and can be added. (​https://groups.google.com/forum/?nomobile=true#!topic/django-developers/guVTzO3RhUs) I'll prepare patch for review.
I guess the model field option could be called db_column_comment. I closed #28407 (introspecting column comments) as a duplicate since that should be implemented as part of this.
Replying to Tim Graham: I guess the model field option could be called db_column_comment. I closed #28407 (introspecting column comments) as a duplicate since that should be implemented as part of this. I think that we don't need new param, because comment for django admin may be useful to store in database. But I can't decide on implementation, can you give me an advice? Postgres and oracle have a syntax like comment on {table}.{column} for storing comments, so this needs to be done after table/column creation, so there are two ways: Add it to post migrate signal, as for content types. But I can implement it as a third-party lib Add this SQL after database creation in schema.py Which way is better?
I'm not sure what you mean by "comment for django admin". There isn't an existing option with that name. As for the implementation, the second approach sounds better.
Replying to Ivan Chernoff: After discussion on mailing list, the feature is good and can be added. (​https://groups.google.com/forum/?nomobile=true#!topic/django-developers/guVTzO3RhUs) I'll prepare patch for review. Any news on this patch? It looks like there is a green light for one, and Tim already answered about the preferable approach. Do you need any help on making this hapen? It would be a fantastic feature for Django!
Replying to Rodrigo Silva: Replying to Ivan Chernoff: After discussion on mailing list, the feature is good and can be added. (​https://groups.google.com/forum/?nomobile=true#!topic/django-developers/guVTzO3RhUs) I'll prepare patch for review. Any news on this patch? It looks like there is a green light for one, and Tim already answered about the preferable approach. Do you need any help on making this hapen? It would be a fantastic feature for Django! I've a small library based on contenttype design: on post-migrate it copies all comments to database (PostgreSQL only). ​https://github.com/vanadium23/django-db-comments
​PR
i assign it to me i will close this pullRequest caused by imperfection ​https://github.com/django/django/pull/12605 I will bring a new PullRequest as soon as possible. in at least a week
Adding a new setting is always a bit controversial. This should be discussed first on DevelopersMailingList. Personally I don't think that we need it.
i agree. your opinion. i remove new settings. ENABLE_DB_COMMENT_WITH_HELP_TEXT AND i added few a more commit for support postgresql oracle
here is mail list for this feature ​https://groups.google.com/forum/?nomobile=true#!topic/django-developers/guVTzO3RhUs
If you don't mind me asking... could you give me a feedback to this feature?
Hi felixxm I want this feature to be reflected in Django.3.1 I know this feature that many people won't use. but some people who communicate with DBA would be helpful this feature sometimes ORM take control from DBA like these situation... DBAs: we want to know comment(help_text) not to *.py could you use COMMENT SQL in migrationFile? Developers: No, It's Impossible because django ORM does not support COMMENT SQL DBAs: ???? that means .. should we refer *.py file everytimes even though COMMENT SQL exists but can't use? DBAs: it's illogical!!!! Developers: but if we use COMMENT SQL, we customize everytimes to generate migrate files Developers: it's inefficient!!!! That's one of the reasons for avoiding ORM. If you don't mind me asking... could you give me a feedback to this feature?
This patch is not ready for a review. Docs and tests are still missing.
Rebased a previous PR to add support to inspectdb here: ​https://github.com/django/django/pull/13737 Still missing some tests and it looks like we would like it to be a verbose_name field and not a comment
Owner updated based on suggestion in django-developers group ​https://groups.google.com/g/django-developers/c/guVTzO3RhUs
Status update: KimSoungRyoul has created a new PR (​here) that addresses some of the suggestions raised by atombrella in May 2020 on the previous PR (​here). Huge thanks to KimSoungRyoul for resuming work on this feature! The new PR is not complete yet. Still needed: 1) Test coverage. 2) Additional changes to the proposed docs including further explanation, a warning about lack of SQLite support, and a warning about overwriting comments made by DBAs in the db, versionadded annotation, and a couple other things. 3) Release note. 4) General code review, especially related to where sqlite warnings should be triggered in the code to avoid having hard-coded vendor names in django/db/models/base.py. Next steps: I'll add some suggestions directly to the new PR and post updates here as this PR progresses.
Update: This feature is still being actively developed. KimSoungRyoul has been leading the development, and we've received input as well from knyghty on the PR. Docs and unit tests have been written, and right now we are discussing edge cases that need to be properly handled or explicitly "unsupported" (such as, for example, Django model classes or fields that Django doesn't actually create in the db such as M2M fields, proxy models, abstract base classes, and more).
This patch is ready for review. It meets everything on the checklist. I'm updating the status so it shows up in Patches Needing Review on the Development Dashboard. If you would like additional information about some of the design and implementation decisions behind this patch, please refer to the Github Pull Request conversation at ​https://github.com/django/django/pull/14463
Update: This patch has been reviewed (refer to ​this comment on the Github PR for the details on the patch review checklist for this patch). I have marked it "Ready for checkin" Note: Although I am the owner of this ticket, I am NOT the author of this patch. The author of the patch is KimSoungRyoul. Because I am not the author, I conducted the patch review using ​the patch review checklist.
May I change assignee to me? perhaps is there any problem?
Status update: KSR has updated the patch on github ​https://github.com/django/django/pull/14463 and there's continued discussion there, including additional changes proposed by other community members...
```

## Patch

```diff
diff --git a/django/core/management/commands/inspectdb.py b/django/core/management/commands/inspectdb.py
--- a/django/core/management/commands/inspectdb.py
+++ b/django/core/management/commands/inspectdb.py
@@ -78,18 +78,16 @@ def table2model(table_name):
             )
             yield "from %s import models" % self.db_module
             known_models = []
-            table_info = connection.introspection.get_table_list(cursor)
-
             # Determine types of tables and/or views to be introspected.
             types = {"t"}
             if options["include_partitions"]:
                 types.add("p")
             if options["include_views"]:
                 types.add("v")
+            table_info = connection.introspection.get_table_list(cursor)
+            table_info = {info.name: info for info in table_info if info.type in types}
 
-            for table_name in options["table"] or sorted(
-                info.name for info in table_info if info.type in types
-            ):
+            for table_name in options["table"] or sorted(name for name in table_info):
                 if table_name_filter is not None and callable(table_name_filter):
                     if not table_name_filter(table_name):
                         continue
@@ -232,6 +230,10 @@ def table2model(table_name):
                     if field_type.startswith(("ForeignKey(", "OneToOneField(")):
                         field_desc += ", models.DO_NOTHING"
 
+                    # Add comment.
+                    if connection.features.supports_comments and row.comment:
+                        extra_params["db_comment"] = row.comment
+
                     if extra_params:
                         if not field_desc.endswith("("):
                             field_desc += ", "
@@ -242,14 +244,22 @@ def table2model(table_name):
                     if comment_notes:
                         field_desc += "  # " + " ".join(comment_notes)
                     yield "    %s" % field_desc
-                is_view = any(
-                    info.name == table_name and info.type == "v" for info in table_info
-                )
-                is_partition = any(
-                    info.name == table_name and info.type == "p" for info in table_info
-                )
+                comment = None
+                if info := table_info.get(table_name):
+                    is_view = info.type == "v"
+                    is_partition = info.type == "p"
+                    if connection.features.supports_comments:
+                        comment = info.comment
+                else:
+                    is_view = False
+                    is_partition = False
                 yield from self.get_meta(
-                    table_name, constraints, column_to_field_name, is_view, is_partition
+                    table_name,
+                    constraints,
+                    column_to_field_name,
+                    is_view,
+                    is_partition,
+                    comment,
                 )
 
     def normalize_col_name(self, col_name, used_column_names, is_relation):
@@ -353,7 +363,13 @@ def get_field_type(self, connection, table_name, row):
         return field_type, field_params, field_notes
 
     def get_meta(
-        self, table_name, constraints, column_to_field_name, is_view, is_partition
+        self,
+        table_name,
+        constraints,
+        column_to_field_name,
+        is_view,
+        is_partition,
+        comment,
     ):
         """
         Return a sequence comprising the lines of code necessary
@@ -391,4 +407,6 @@ def get_meta(
         if unique_together:
             tup = "(" + ", ".join(unique_together) + ",)"
             meta += ["        unique_together = %s" % tup]
+        if comment:
+            meta += [f"        db_table_comment = {comment!r}"]
         return meta
diff --git a/django/db/backends/base/features.py b/django/db/backends/base/features.py
--- a/django/db/backends/base/features.py
+++ b/django/db/backends/base/features.py
@@ -334,6 +334,11 @@ class BaseDatabaseFeatures:
     # Does the backend support non-deterministic collations?
     supports_non_deterministic_collations = True
 
+    # Does the backend support column and table comments?
+    supports_comments = False
+    # Does the backend support column comments in ADD COLUMN statements?
+    supports_comments_inline = False
+
     # Does the backend support the logical XOR operator?
     supports_logical_xor = False
 
diff --git a/django/db/backends/base/schema.py b/django/db/backends/base/schema.py
--- a/django/db/backends/base/schema.py
+++ b/django/db/backends/base/schema.py
@@ -141,6 +141,9 @@ class BaseDatabaseSchemaEditor:
 
     sql_delete_procedure = "DROP PROCEDURE %(procedure)s"
 
+    sql_alter_table_comment = "COMMENT ON TABLE %(table)s IS %(comment)s"
+    sql_alter_column_comment = "COMMENT ON COLUMN %(table)s.%(column)s IS %(comment)s"
+
     def __init__(self, connection, collect_sql=False, atomic=True):
         self.connection = connection
         self.collect_sql = collect_sql
@@ -289,6 +292,8 @@ def _iter_column_sql(
         yield column_db_type
         if collation := field_db_params.get("collation"):
             yield self._collate_sql(collation)
+        if self.connection.features.supports_comments_inline and field.db_comment:
+            yield self._comment_sql(field.db_comment)
         # Work out nullability.
         null = field.null
         # Include a default value, if requested.
@@ -445,6 +450,23 @@ def create_model(self, model):
         # definition.
         self.execute(sql, params or None)
 
+        if self.connection.features.supports_comments:
+            # Add table comment.
+            if model._meta.db_table_comment:
+                self.alter_db_table_comment(model, None, model._meta.db_table_comment)
+            # Add column comments.
+            if not self.connection.features.supports_comments_inline:
+                for field in model._meta.local_fields:
+                    if field.db_comment:
+                        field_db_params = field.db_parameters(
+                            connection=self.connection
+                        )
+                        field_type = field_db_params["type"]
+                        self.execute(
+                            *self._alter_column_comment_sql(
+                                model, field, field_type, field.db_comment
+                            )
+                        )
         # Add any field index and index_together's (deferred as SQLite
         # _remake_table needs it).
         self.deferred_sql.extend(self._model_indexes_sql(model))
@@ -614,6 +636,15 @@ def alter_db_table(self, model, old_db_table, new_db_table):
             if isinstance(sql, Statement):
                 sql.rename_table_references(old_db_table, new_db_table)
 
+    def alter_db_table_comment(self, model, old_db_table_comment, new_db_table_comment):
+        self.execute(
+            self.sql_alter_table_comment
+            % {
+                "table": self.quote_name(model._meta.db_table),
+                "comment": self.quote_value(new_db_table_comment or ""),
+            }
+        )
+
     def alter_db_tablespace(self, model, old_db_tablespace, new_db_tablespace):
         """Move a model's table between tablespaces."""
         self.execute(
@@ -693,6 +724,18 @@ def add_field(self, model, field):
                 "changes": changes_sql,
             }
             self.execute(sql, params)
+        # Add field comment, if required.
+        if (
+            field.db_comment
+            and self.connection.features.supports_comments
+            and not self.connection.features.supports_comments_inline
+        ):
+            field_type = db_params["type"]
+            self.execute(
+                *self._alter_column_comment_sql(
+                    model, field, field_type, field.db_comment
+                )
+            )
         # Add an index, if required
         self.deferred_sql.extend(self._field_indexes_sql(model, field))
         # Reset connection if required
@@ -813,6 +856,11 @@ def _alter_field(
             self.connection.features.supports_foreign_keys
             and old_field.remote_field
             and old_field.db_constraint
+            and self._field_should_be_altered(
+                old_field,
+                new_field,
+                ignore={"db_comment"},
+            )
         ):
             fk_names = self._constraint_names(
                 model, [old_field.column], foreign_key=True
@@ -949,11 +997,15 @@ def _alter_field(
         # Type suffix change? (e.g. auto increment).
         old_type_suffix = old_field.db_type_suffix(connection=self.connection)
         new_type_suffix = new_field.db_type_suffix(connection=self.connection)
-        # Type or collation change?
+        # Type, collation, or comment change?
         if (
             old_type != new_type
             or old_type_suffix != new_type_suffix
             or old_collation != new_collation
+            or (
+                self.connection.features.supports_comments
+                and old_field.db_comment != new_field.db_comment
+            )
         ):
             fragment, other_actions = self._alter_column_type_sql(
                 model, old_field, new_field, new_type, old_collation, new_collation
@@ -1211,12 +1263,26 @@ def _alter_column_type_sql(
         an ALTER TABLE statement and a list of extra (sql, params) tuples to
         run once the field is altered.
         """
+        other_actions = []
         if collate_sql := self._collate_sql(
             new_collation, old_collation, model._meta.db_table
         ):
             collate_sql = f" {collate_sql}"
         else:
             collate_sql = ""
+        # Comment change?
+        comment_sql = ""
+        if self.connection.features.supports_comments and not new_field.many_to_many:
+            if old_field.db_comment != new_field.db_comment:
+                # PostgreSQL and Oracle can't execute 'ALTER COLUMN ...' and
+                # 'COMMENT ON ...' at the same time.
+                sql, params = self._alter_column_comment_sql(
+                    model, new_field, new_type, new_field.db_comment
+                )
+                if sql:
+                    other_actions.append((sql, params))
+            if new_field.db_comment:
+                comment_sql = self._comment_sql(new_field.db_comment)
         return (
             (
                 self.sql_alter_column_type
@@ -1224,12 +1290,27 @@ def _alter_column_type_sql(
                     "column": self.quote_name(new_field.column),
                     "type": new_type,
                     "collation": collate_sql,
+                    "comment": comment_sql,
                 },
                 [],
             ),
+            other_actions,
+        )
+
+    def _alter_column_comment_sql(self, model, new_field, new_type, new_db_comment):
+        return (
+            self.sql_alter_column_comment
+            % {
+                "table": self.quote_name(model._meta.db_table),
+                "column": self.quote_name(new_field.column),
+                "comment": self._comment_sql(new_db_comment),
+            },
             [],
         )
 
+    def _comment_sql(self, comment):
+        return self.quote_value(comment or "")
+
     def _alter_many_to_many(self, model, old_field, new_field, strict):
         """Alter M2Ms to repoint their to= endpoints."""
         # Rename the through table
@@ -1423,16 +1504,18 @@ def _field_indexes_sql(self, model, field):
             output.append(self._create_index_sql(model, fields=[field]))
         return output
 
-    def _field_should_be_altered(self, old_field, new_field):
+    def _field_should_be_altered(self, old_field, new_field, ignore=None):
+        ignore = ignore or set()
         _, old_path, old_args, old_kwargs = old_field.deconstruct()
         _, new_path, new_args, new_kwargs = new_field.deconstruct()
         # Don't alter when:
         # - changing only a field name
         # - changing an attribute that doesn't affect the schema
+        # - changing an attribute in the provided set of ignored attributes
         # - adding only a db_column and the column name is not changed
-        for attr in old_field.non_db_attrs:
+        for attr in ignore.union(old_field.non_db_attrs):
             old_kwargs.pop(attr, None)
-        for attr in new_field.non_db_attrs:
+        for attr in ignore.union(new_field.non_db_attrs):
             new_kwargs.pop(attr, None)
         return self.quote_name(old_field.column) != self.quote_name(
             new_field.column
diff --git a/django/db/backends/mysql/features.py b/django/db/backends/mysql/features.py
--- a/django/db/backends/mysql/features.py
+++ b/django/db/backends/mysql/features.py
@@ -18,6 +18,8 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     requires_explicit_null_ordering_when_grouping = True
     atomic_transactions = False
     can_clone_databases = True
+    supports_comments = True
+    supports_comments_inline = True
     supports_temporal_subtraction = True
     supports_slicing_ordering_in_compound = True
     supports_index_on_text_field = False
diff --git a/django/db/backends/mysql/introspection.py b/django/db/backends/mysql/introspection.py
--- a/django/db/backends/mysql/introspection.py
+++ b/django/db/backends/mysql/introspection.py
@@ -5,18 +5,20 @@
 
 from django.db.backends.base.introspection import BaseDatabaseIntrospection
 from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
-from django.db.backends.base.introspection import TableInfo
+from django.db.backends.base.introspection import TableInfo as BaseTableInfo
 from django.db.models import Index
 from django.utils.datastructures import OrderedSet
 
 FieldInfo = namedtuple(
-    "FieldInfo", BaseFieldInfo._fields + ("extra", "is_unsigned", "has_json_constraint")
+    "FieldInfo",
+    BaseFieldInfo._fields + ("extra", "is_unsigned", "has_json_constraint", "comment"),
 )
 InfoLine = namedtuple(
     "InfoLine",
     "col_name data_type max_len num_prec num_scale extra column_default "
-    "collation is_unsigned",
+    "collation is_unsigned comment",
 )
+TableInfo = namedtuple("TableInfo", BaseTableInfo._fields + ("comment",))
 
 
 class DatabaseIntrospection(BaseDatabaseIntrospection):
@@ -68,9 +70,18 @@ def get_field_type(self, data_type, description):
 
     def get_table_list(self, cursor):
         """Return a list of table and view names in the current database."""
-        cursor.execute("SHOW FULL TABLES")
+        cursor.execute(
+            """
+            SELECT
+                table_name,
+                table_type,
+                table_comment
+            FROM information_schema.tables
+            WHERE table_schema = DATABASE()
+            """
+        )
         return [
-            TableInfo(row[0], {"BASE TABLE": "t", "VIEW": "v"}.get(row[1]))
+            TableInfo(row[0], {"BASE TABLE": "t", "VIEW": "v"}.get(row[1]), row[2])
             for row in cursor.fetchall()
         ]
 
@@ -128,7 +139,8 @@ def get_table_description(self, cursor, table_name):
                 CASE
                     WHEN column_type LIKE '%% unsigned' THEN 1
                     ELSE 0
-                END AS is_unsigned
+                END AS is_unsigned,
+                column_comment
             FROM information_schema.columns
             WHERE table_name = %s AND table_schema = DATABASE()
             """,
@@ -159,6 +171,7 @@ def to_int(i):
                     info.extra,
                     info.is_unsigned,
                     line[0] in json_constraints,
+                    info.comment,
                 )
             )
         return fields
diff --git a/django/db/backends/mysql/schema.py b/django/db/backends/mysql/schema.py
--- a/django/db/backends/mysql/schema.py
+++ b/django/db/backends/mysql/schema.py
@@ -9,7 +9,7 @@ class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):
 
     sql_alter_column_null = "MODIFY %(column)s %(type)s NULL"
     sql_alter_column_not_null = "MODIFY %(column)s %(type)s NOT NULL"
-    sql_alter_column_type = "MODIFY %(column)s %(type)s%(collation)s"
+    sql_alter_column_type = "MODIFY %(column)s %(type)s%(collation)s%(comment)s"
     sql_alter_column_no_default_null = "ALTER COLUMN %(column)s SET DEFAULT NULL"
 
     # No 'CASCADE' which works as a no-op in MySQL but is undocumented
@@ -32,6 +32,9 @@ class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):
 
     sql_create_index = "CREATE INDEX %(name)s ON %(table)s (%(columns)s)%(extra)s"
 
+    sql_alter_table_comment = "ALTER TABLE %(table)s COMMENT = %(comment)s"
+    sql_alter_column_comment = None
+
     @property
     def sql_delete_check(self):
         if self.connection.mysql_is_mariadb:
@@ -228,3 +231,11 @@ def _alter_column_type_sql(
     def _rename_field_sql(self, table, old_field, new_field, new_type):
         new_type = self._set_field_new_type_null_status(old_field, new_type)
         return super()._rename_field_sql(table, old_field, new_field, new_type)
+
+    def _alter_column_comment_sql(self, model, new_field, new_type, new_db_comment):
+        # Comment is alter when altering the column type.
+        return "", []
+
+    def _comment_sql(self, comment):
+        comment_sql = super()._comment_sql(comment)
+        return f" COMMENT {comment_sql}"
diff --git a/django/db/backends/oracle/features.py b/django/db/backends/oracle/features.py
--- a/django/db/backends/oracle/features.py
+++ b/django/db/backends/oracle/features.py
@@ -25,6 +25,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     supports_partially_nullable_unique_constraints = False
     supports_deferrable_unique_constraints = True
     truncates_names = True
+    supports_comments = True
     supports_tablespaces = True
     supports_sequence_reset = False
     can_introspect_materialized_views = True
diff --git a/django/db/backends/oracle/introspection.py b/django/db/backends/oracle/introspection.py
--- a/django/db/backends/oracle/introspection.py
+++ b/django/db/backends/oracle/introspection.py
@@ -5,10 +5,13 @@
 from django.db import models
 from django.db.backends.base.introspection import BaseDatabaseIntrospection
 from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
-from django.db.backends.base.introspection import TableInfo
+from django.db.backends.base.introspection import TableInfo as BaseTableInfo
 from django.utils.functional import cached_property
 
-FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield", "is_json"))
+FieldInfo = namedtuple(
+    "FieldInfo", BaseFieldInfo._fields + ("is_autofield", "is_json", "comment")
+)
+TableInfo = namedtuple("TableInfo", BaseTableInfo._fields + ("comment",))
 
 
 class DatabaseIntrospection(BaseDatabaseIntrospection):
@@ -77,8 +80,14 @@ def get_table_list(self, cursor):
         """Return a list of table and view names in the current database."""
         cursor.execute(
             """
-            SELECT table_name, 't'
+            SELECT
+                user_tables.table_name,
+                't',
+                user_tab_comments.comments
             FROM user_tables
+            LEFT OUTER JOIN
+                user_tab_comments
+                ON user_tab_comments.table_name = user_tables.table_name
             WHERE
                 NOT EXISTS (
                     SELECT 1
@@ -86,13 +95,13 @@ def get_table_list(self, cursor):
                     WHERE user_mviews.mview_name = user_tables.table_name
                 )
             UNION ALL
-            SELECT view_name, 'v' FROM user_views
+            SELECT view_name, 'v', NULL FROM user_views
             UNION ALL
-            SELECT mview_name, 'v' FROM user_mviews
+            SELECT mview_name, 'v', NULL FROM user_mviews
         """
         )
         return [
-            TableInfo(self.identifier_converter(row[0]), row[1])
+            TableInfo(self.identifier_converter(row[0]), row[1], row[2])
             for row in cursor.fetchall()
         ]
 
@@ -131,10 +140,15 @@ def get_table_description(self, cursor, table_name):
                     )
                     THEN 1
                     ELSE 0
-                END as is_json
+                END as is_json,
+                user_col_comments.comments as col_comment
             FROM user_tab_cols
             LEFT OUTER JOIN
                 user_tables ON user_tables.table_name = user_tab_cols.table_name
+            LEFT OUTER JOIN
+                user_col_comments ON
+                user_col_comments.column_name = user_tab_cols.column_name AND
+                user_col_comments.table_name = user_tab_cols.table_name
             WHERE user_tab_cols.table_name = UPPER(%s)
             """,
             [table_name],
@@ -146,6 +160,7 @@ def get_table_description(self, cursor, table_name):
                 collation,
                 is_autofield,
                 is_json,
+                comment,
             )
             for (
                 column,
@@ -154,6 +169,7 @@ def get_table_description(self, cursor, table_name):
                 display_size,
                 is_autofield,
                 is_json,
+                comment,
             ) in cursor.fetchall()
         }
         self.cache_bust_counter += 1
@@ -165,7 +181,14 @@ def get_table_description(self, cursor, table_name):
         description = []
         for desc in cursor.description:
             name = desc[0]
-            display_size, default, collation, is_autofield, is_json = field_map[name]
+            (
+                display_size,
+                default,
+                collation,
+                is_autofield,
+                is_json,
+                comment,
+            ) = field_map[name]
             name %= {}  # cx_Oracle, for some reason, doubles percent signs.
             description.append(
                 FieldInfo(
@@ -180,6 +203,7 @@ def get_table_description(self, cursor, table_name):
                     collation,
                     is_autofield,
                     is_json,
+                    comment,
                 )
             )
         return description
diff --git a/django/db/backends/postgresql/features.py b/django/db/backends/postgresql/features.py
--- a/django/db/backends/postgresql/features.py
+++ b/django/db/backends/postgresql/features.py
@@ -22,6 +22,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     has_select_for_update_skip_locked = True
     has_select_for_no_key_update = True
     can_release_savepoints = True
+    supports_comments = True
     supports_tablespaces = True
     supports_transactions = True
     can_introspect_materialized_views = True
diff --git a/django/db/backends/postgresql/introspection.py b/django/db/backends/postgresql/introspection.py
--- a/django/db/backends/postgresql/introspection.py
+++ b/django/db/backends/postgresql/introspection.py
@@ -2,10 +2,11 @@
 
 from django.db.backends.base.introspection import BaseDatabaseIntrospection
 from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
-from django.db.backends.base.introspection import TableInfo
+from django.db.backends.base.introspection import TableInfo as BaseTableInfo
 from django.db.models import Index
 
-FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield",))
+FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield", "comment"))
+TableInfo = namedtuple("TableInfo", BaseTableInfo._fields + ("comment",))
 
 
 class DatabaseIntrospection(BaseDatabaseIntrospection):
@@ -62,7 +63,8 @@ def get_table_list(self, cursor):
                     WHEN c.relispartition THEN 'p'
                     WHEN c.relkind IN ('m', 'v') THEN 'v'
                     ELSE 't'
-                END
+                END,
+                obj_description(c.oid)
             FROM pg_catalog.pg_class c
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
             WHERE c.relkind IN ('f', 'm', 'p', 'r', 'v')
@@ -91,7 +93,8 @@ def get_table_description(self, cursor, table_name):
                 NOT (a.attnotnull OR (t.typtype = 'd' AND t.typnotnull)) AS is_nullable,
                 pg_get_expr(ad.adbin, ad.adrelid) AS column_default,
                 CASE WHEN collname = 'default' THEN NULL ELSE collname END AS collation,
-                a.attidentity != '' AS is_autofield
+                a.attidentity != '' AS is_autofield,
+                col_description(a.attrelid, a.attnum) AS column_comment
             FROM pg_attribute a
             LEFT JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
             LEFT JOIN pg_collation co ON a.attcollation = co.oid
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -170,6 +170,7 @@ def _detect_changes(self, convert_apps=None, graph=None):
         self.generate_created_proxies()
         self.generate_altered_options()
         self.generate_altered_managers()
+        self.generate_altered_db_table_comment()
 
         # Create the renamed fields and store them in self.renamed_fields.
         # They are used by create_altered_indexes(), generate_altered_fields(),
@@ -1552,6 +1553,28 @@ def generate_altered_db_table(self):
                     ),
                 )
 
+    def generate_altered_db_table_comment(self):
+        models_to_check = self.kept_model_keys.union(
+            self.kept_proxy_keys, self.kept_unmanaged_keys
+        )
+        for app_label, model_name in sorted(models_to_check):
+            old_model_name = self.renamed_models.get(
+                (app_label, model_name), model_name
+            )
+            old_model_state = self.from_state.models[app_label, old_model_name]
+            new_model_state = self.to_state.models[app_label, model_name]
+
+            old_db_table_comment = old_model_state.options.get("db_table_comment")
+            new_db_table_comment = new_model_state.options.get("db_table_comment")
+            if old_db_table_comment != new_db_table_comment:
+                self.add_operation(
+                    app_label,
+                    operations.AlterModelTableComment(
+                        name=model_name,
+                        table_comment=new_db_table_comment,
+                    ),
+                )
+
     def generate_altered_options(self):
         """
         Work out if any non-schema-affecting options have changed and make an
diff --git a/django/db/migrations/operations/__init__.py b/django/db/migrations/operations/__init__.py
--- a/django/db/migrations/operations/__init__.py
+++ b/django/db/migrations/operations/__init__.py
@@ -6,6 +6,7 @@
     AlterModelManagers,
     AlterModelOptions,
     AlterModelTable,
+    AlterModelTableComment,
     AlterOrderWithRespectTo,
     AlterUniqueTogether,
     CreateModel,
@@ -21,6 +22,7 @@
     "CreateModel",
     "DeleteModel",
     "AlterModelTable",
+    "AlterModelTableComment",
     "AlterUniqueTogether",
     "RenameModel",
     "AlterIndexTogether",
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -529,6 +529,44 @@ def migration_name_fragment(self):
         return "alter_%s_table" % self.name_lower
 
 
+class AlterModelTableComment(ModelOptionOperation):
+    def __init__(self, name, table_comment):
+        self.table_comment = table_comment
+        super().__init__(name)
+
+    def deconstruct(self):
+        kwargs = {
+            "name": self.name,
+            "table_comment": self.table_comment,
+        }
+        return (self.__class__.__qualname__, [], kwargs)
+
+    def state_forwards(self, app_label, state):
+        state.alter_model_options(
+            app_label, self.name_lower, {"db_table_comment": self.table_comment}
+        )
+
+    def database_forwards(self, app_label, schema_editor, from_state, to_state):
+        new_model = to_state.apps.get_model(app_label, self.name)
+        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
+            old_model = from_state.apps.get_model(app_label, self.name)
+            schema_editor.alter_db_table_comment(
+                new_model,
+                old_model._meta.db_table_comment,
+                new_model._meta.db_table_comment,
+            )
+
+    def database_backwards(self, app_label, schema_editor, from_state, to_state):
+        return self.database_forwards(app_label, schema_editor, from_state, to_state)
+
+    def describe(self):
+        return f"Alter {self.name} table comment"
+
+    @property
+    def migration_name_fragment(self):
+        return f"alter_{self.name_lower}_table_comment"
+
+
 class AlterTogetherOptionOperation(ModelOptionOperation):
     option_name = None
 
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -1556,6 +1556,7 @@ def check(cls, **kwargs):
                 *cls._check_ordering(),
                 *cls._check_constraints(databases),
                 *cls._check_default_pk(),
+                *cls._check_db_table_comment(databases),
             ]
 
         return errors
@@ -1592,6 +1593,29 @@ def _check_default_pk(cls):
             ]
         return []
 
+    @classmethod
+    def _check_db_table_comment(cls, databases):
+        if not cls._meta.db_table_comment:
+            return []
+        errors = []
+        for db in databases:
+            if not router.allow_migrate_model(db, cls):
+                continue
+            connection = connections[db]
+            if not (
+                connection.features.supports_comments
+                or "supports_comments" in cls._meta.required_db_features
+            ):
+                errors.append(
+                    checks.Warning(
+                        f"{connection.display_name} does not support comments on "
+                        f"tables (db_table_comment).",
+                        obj=cls,
+                        id="models.W046",
+                    )
+                )
+        return errors
+
     @classmethod
     def _check_swappable(cls):
         """Check if the swapped model exists."""
diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -200,6 +200,7 @@ def __init__(
         auto_created=False,
         validators=(),
         error_messages=None,
+        db_comment=None,
     ):
         self.name = name
         self.verbose_name = verbose_name  # May be set by set_attributes_from_name
@@ -221,6 +222,7 @@ def __init__(
         self.help_text = help_text
         self.db_index = db_index
         self.db_column = db_column
+        self.db_comment = db_comment
         self._db_tablespace = db_tablespace
         self.auto_created = auto_created
 
@@ -259,6 +261,7 @@ def check(self, **kwargs):
             *self._check_field_name(),
             *self._check_choices(),
             *self._check_db_index(),
+            *self._check_db_comment(**kwargs),
             *self._check_null_allowed_for_primary_keys(),
             *self._check_backend_specific_checks(**kwargs),
             *self._check_validators(),
@@ -385,6 +388,28 @@ def _check_db_index(self):
         else:
             return []
 
+    def _check_db_comment(self, databases=None, **kwargs):
+        if not self.db_comment or not databases:
+            return []
+        errors = []
+        for db in databases:
+            if not router.allow_migrate_model(db, self.model):
+                continue
+            connection = connections[db]
+            if not (
+                connection.features.supports_comments
+                or "supports_comments" in self.model._meta.required_db_features
+            ):
+                errors.append(
+                    checks.Warning(
+                        f"{connection.display_name} does not support comments on "
+                        f"columns (db_comment).",
+                        obj=self,
+                        id="fields.W163",
+                    )
+                )
+        return errors
+
     def _check_null_allowed_for_primary_keys(self):
         if (
             self.primary_key
@@ -538,6 +563,7 @@ def deconstruct(self):
             "choices": None,
             "help_text": "",
             "db_column": None,
+            "db_comment": None,
             "db_tablespace": None,
             "auto_created": False,
             "validators": [],
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -1428,6 +1428,14 @@ def _check_ignored_options(self, **kwargs):
                     id="fields.W345",
                 )
             )
+        if self.db_comment:
+            warnings.append(
+                checks.Warning(
+                    "db_comment has no effect on ManyToManyField.",
+                    obj=self,
+                    id="fields.W346",
+                )
+            )
 
         return warnings
 
diff --git a/django/db/models/options.py b/django/db/models/options.py
--- a/django/db/models/options.py
+++ b/django/db/models/options.py
@@ -30,6 +30,7 @@
     "verbose_name",
     "verbose_name_plural",
     "db_table",
+    "db_table_comment",
     "ordering",
     "unique_together",
     "permissions",
@@ -112,6 +113,7 @@ def __init__(self, meta, app_label=None):
         self.verbose_name = None
         self.verbose_name_plural = None
         self.db_table = ""
+        self.db_table_comment = ""
         self.ordering = []
         self._ordering_clash = False
         self.indexes = []

```

## Test Patch

```diff
diff --git a/tests/inspectdb/models.py b/tests/inspectdb/models.py
--- a/tests/inspectdb/models.py
+++ b/tests/inspectdb/models.py
@@ -132,3 +132,11 @@ class Meta:
             )
         ]
         required_db_features = {"supports_expression_indexes"}
+
+
+class DbComment(models.Model):
+    rank = models.IntegerField(db_comment="'Rank' column comment")
+
+    class Meta:
+        db_table_comment = "Custom table comment"
+        required_db_features = {"supports_comments"}
diff --git a/tests/inspectdb/tests.py b/tests/inspectdb/tests.py
--- a/tests/inspectdb/tests.py
+++ b/tests/inspectdb/tests.py
@@ -129,6 +129,24 @@ def test_json_field(self):
             "null_json_field = models.JSONField(blank=True, null=True)", output
         )
 
+    @skipUnlessDBFeature("supports_comments")
+    def test_db_comments(self):
+        out = StringIO()
+        call_command("inspectdb", "inspectdb_dbcomment", stdout=out)
+        output = out.getvalue()
+        integer_field_type = connection.features.introspected_field_types[
+            "IntegerField"
+        ]
+        self.assertIn(
+            f"rank = models.{integer_field_type}("
+            f"db_comment=\"'Rank' column comment\")",
+            output,
+        )
+        self.assertIn(
+            "        db_table_comment = 'Custom table comment'",
+            output,
+        )
+
     @skipUnlessDBFeature("supports_collation_on_charfield")
     @skipUnless(test_collation, "Language collations are not supported.")
     def test_char_field_db_collation(self):
diff --git a/tests/introspection/models.py b/tests/introspection/models.py
--- a/tests/introspection/models.py
+++ b/tests/introspection/models.py
@@ -102,3 +102,11 @@ class Meta:
                 condition=models.Q(color__isnull=True),
             ),
         ]
+
+
+class DbCommentModel(models.Model):
+    name = models.CharField(max_length=15, db_comment="'Name' column comment")
+
+    class Meta:
+        db_table_comment = "Custom table comment"
+        required_db_features = {"supports_comments"}
diff --git a/tests/introspection/tests.py b/tests/introspection/tests.py
--- a/tests/introspection/tests.py
+++ b/tests/introspection/tests.py
@@ -9,6 +9,7 @@
     City,
     Comment,
     Country,
+    DbCommentModel,
     District,
     Reporter,
     UniqueConstraintConditionModel,
@@ -179,6 +180,26 @@ def test_smallautofield(self):
             [connection.introspection.get_field_type(r[1], r) for r in desc],
         )
 
+    @skipUnlessDBFeature("supports_comments")
+    def test_db_comments(self):
+        with connection.cursor() as cursor:
+            desc = connection.introspection.get_table_description(
+                cursor, DbCommentModel._meta.db_table
+            )
+            table_list = connection.introspection.get_table_list(cursor)
+        self.assertEqual(
+            ["'Name' column comment"],
+            [field.comment for field in desc if field.name == "name"],
+        )
+        self.assertEqual(
+            ["Custom table comment"],
+            [
+                table.comment
+                for table in table_list
+                if table.name == "introspection_dbcommentmodel"
+            ],
+        )
+
     # Regression test for #9991 - 'real' types in postgres
     @skipUnlessDBFeature("has_real_datatype")
     def test_postgresql_real_type(self):
diff --git a/tests/invalid_models_tests/test_models.py b/tests/invalid_models_tests/test_models.py
--- a/tests/invalid_models_tests/test_models.py
+++ b/tests/invalid_models_tests/test_models.py
@@ -1872,6 +1872,37 @@ def dummy_function(*args, **kwargs):
         )
 
 
+@isolate_apps("invalid_models_tests")
+class DbTableCommentTests(TestCase):
+    def test_db_table_comment(self):
+        class Model(models.Model):
+            class Meta:
+                db_table_comment = "Table comment"
+
+        errors = Model.check(databases=self.databases)
+        expected = (
+            []
+            if connection.features.supports_comments
+            else [
+                Warning(
+                    f"{connection.display_name} does not support comments on tables "
+                    f"(db_table_comment).",
+                    obj=Model,
+                    id="models.W046",
+                ),
+            ]
+        )
+        self.assertEqual(errors, expected)
+
+    def test_db_table_comment_required_db_features(self):
+        class Model(models.Model):
+            class Meta:
+                db_table_comment = "Table comment"
+                required_db_features = {"supports_comments"}
+
+        self.assertEqual(Model.check(databases=self.databases), [])
+
+
 class MultipleAutoFieldsTests(TestCase):
     def test_multiple_autofields(self):
         msg = (
diff --git a/tests/invalid_models_tests/test_ordinary_fields.py b/tests/invalid_models_tests/test_ordinary_fields.py
--- a/tests/invalid_models_tests/test_ordinary_fields.py
+++ b/tests/invalid_models_tests/test_ordinary_fields.py
@@ -1023,3 +1023,35 @@ class Model(models.Model):
             field = models.JSONField(default=callable_default)
 
         self.assertEqual(Model._meta.get_field("field").check(), [])
+
+
+@isolate_apps("invalid_models_tests")
+class DbCommentTests(TestCase):
+    def test_db_comment(self):
+        class Model(models.Model):
+            field = models.IntegerField(db_comment="Column comment")
+
+        errors = Model._meta.get_field("field").check(databases=self.databases)
+        expected = (
+            []
+            if connection.features.supports_comments
+            else [
+                DjangoWarning(
+                    f"{connection.display_name} does not support comments on columns "
+                    f"(db_comment).",
+                    obj=Model._meta.get_field("field"),
+                    id="fields.W163",
+                ),
+            ]
+        )
+        self.assertEqual(errors, expected)
+
+    def test_db_comment_required_db_features(self):
+        class Model(models.Model):
+            field = models.IntegerField(db_comment="Column comment")
+
+            class Meta:
+                required_db_features = {"supports_comments"}
+
+        errors = Model._meta.get_field("field").check(databases=self.databases)
+        self.assertEqual(errors, [])
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -94,7 +94,9 @@ class Model(models.Model):
             name = models.CharField(max_length=20)
 
         class ModelM2M(models.Model):
-            m2m = models.ManyToManyField(Model, null=True, validators=[lambda x: x])
+            m2m = models.ManyToManyField(
+                Model, null=True, validators=[lambda x: x], db_comment="Column comment"
+            )
 
         field = ModelM2M._meta.get_field("m2m")
         self.assertEqual(
@@ -110,6 +112,11 @@ class ModelM2M(models.Model):
                     obj=field,
                     id="fields.W341",
                 ),
+                DjangoWarning(
+                    "db_comment has no effect on ManyToManyField.",
+                    obj=field,
+                    id="fields.W346",
+                ),
             ],
         )
 
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -773,6 +773,14 @@ class AutodetectorTests(BaseAutodetectorTests):
             "verbose_name": "Authi",
         },
     )
+    author_with_db_table_comment = ModelState(
+        "testapp",
+        "Author",
+        [
+            ("id", models.AutoField(primary_key=True)),
+        ],
+        {"db_table_comment": "Table comment"},
+    )
     author_with_db_table_options = ModelState(
         "testapp",
         "Author",
@@ -2349,6 +2357,58 @@ def test_alter_db_table_with_model_change(self):
             changes, "testapp", 0, 1, name="newauthor", table="author_three"
         )
 
+    def test_alter_db_table_comment_add(self):
+        changes = self.get_changes(
+            [self.author_empty], [self.author_with_db_table_comment]
+        )
+        self.assertNumberMigrations(changes, "testapp", 1)
+        self.assertOperationTypes(changes, "testapp", 0, ["AlterModelTableComment"])
+        self.assertOperationAttributes(
+            changes, "testapp", 0, 0, name="author", table_comment="Table comment"
+        )
+
+    def test_alter_db_table_comment_change(self):
+        author_with_new_db_table_comment = ModelState(
+            "testapp",
+            "Author",
+            [
+                ("id", models.AutoField(primary_key=True)),
+            ],
+            {"db_table_comment": "New table comment"},
+        )
+        changes = self.get_changes(
+            [self.author_with_db_table_comment],
+            [author_with_new_db_table_comment],
+        )
+        self.assertNumberMigrations(changes, "testapp", 1)
+        self.assertOperationTypes(changes, "testapp", 0, ["AlterModelTableComment"])
+        self.assertOperationAttributes(
+            changes,
+            "testapp",
+            0,
+            0,
+            name="author",
+            table_comment="New table comment",
+        )
+
+    def test_alter_db_table_comment_remove(self):
+        changes = self.get_changes(
+            [self.author_with_db_table_comment],
+            [self.author_empty],
+        )
+        self.assertNumberMigrations(changes, "testapp", 1)
+        self.assertOperationTypes(changes, "testapp", 0, ["AlterModelTableComment"])
+        self.assertOperationAttributes(
+            changes, "testapp", 0, 0, name="author", db_table_comment=None
+        )
+
+    def test_alter_db_table_comment_no_changes(self):
+        changes = self.get_changes(
+            [self.author_with_db_table_comment],
+            [self.author_with_db_table_comment],
+        )
+        self.assertNumberMigrations(changes, "testapp", 0)
+
     def test_identical_regex_doesnt_alter(self):
         from_state = ModelState(
             "testapp",
diff --git a/tests/migrations/test_base.py b/tests/migrations/test_base.py
--- a/tests/migrations/test_base.py
+++ b/tests/migrations/test_base.py
@@ -75,6 +75,20 @@ def _get_column_collation(self, table, column, using):
     def assertColumnCollation(self, table, column, collation, using="default"):
         self.assertEqual(self._get_column_collation(table, column, using), collation)
 
+    def _get_table_comment(self, table, using):
+        with connections[using].cursor() as cursor:
+            return next(
+                t.comment
+                for t in connections[using].introspection.get_table_list(cursor)
+                if t.name == table
+            )
+
+    def assertTableComment(self, table, comment, using="default"):
+        self.assertEqual(self._get_table_comment(table, using), comment)
+
+    def assertTableCommentNotExists(self, table, using="default"):
+        self.assertIn(self._get_table_comment(table, using), [None, ""])
+
     def assertIndexExists(
         self, table, columns, value=True, using="default", index_type=None
     ):
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -1922,6 +1922,37 @@ def test_alter_field_add_db_column_noop(self):
                 operation.database_forwards(app_label, editor, new_state, project_state)
         self.assertColumnExists(rider_table, "pony_id")
 
+    @skipUnlessDBFeature("supports_comments")
+    def test_alter_model_table_comment(self):
+        app_label = "test_almotaco"
+        project_state = self.set_up_test_model(app_label)
+        pony_table = f"{app_label}_pony"
+        # Add table comment.
+        operation = migrations.AlterModelTableComment("Pony", "Custom pony comment")
+        self.assertEqual(operation.describe(), "Alter Pony table comment")
+        self.assertEqual(operation.migration_name_fragment, "alter_pony_table_comment")
+        new_state = project_state.clone()
+        operation.state_forwards(app_label, new_state)
+        self.assertEqual(
+            new_state.models[app_label, "pony"].options["db_table_comment"],
+            "Custom pony comment",
+        )
+        self.assertTableCommentNotExists(pony_table)
+        with connection.schema_editor() as editor:
+            operation.database_forwards(app_label, editor, project_state, new_state)
+        self.assertTableComment(pony_table, "Custom pony comment")
+        # Reversal.
+        with connection.schema_editor() as editor:
+            operation.database_backwards(app_label, editor, new_state, project_state)
+        self.assertTableCommentNotExists(pony_table)
+        # Deconstruction.
+        definition = operation.deconstruct()
+        self.assertEqual(definition[0], "AlterModelTableComment")
+        self.assertEqual(definition[1], [])
+        self.assertEqual(
+            definition[2], {"name": "Pony", "table_comment": "Custom pony comment"}
+        )
+
     def test_alter_field_pk(self):
         """
         The AlterField operation on primary keys (things like PostgreSQL's
diff --git a/tests/schema/tests.py b/tests/schema/tests.py
--- a/tests/schema/tests.py
+++ b/tests/schema/tests.py
@@ -273,6 +273,27 @@ def get_column_collation(self, table, column):
                 if f.name == column
             )
 
+    def get_column_comment(self, table, column):
+        with connection.cursor() as cursor:
+            return next(
+                f.comment
+                for f in connection.introspection.get_table_description(cursor, table)
+                if f.name == column
+            )
+
+    def get_table_comment(self, table):
+        with connection.cursor() as cursor:
+            return next(
+                t.comment
+                for t in connection.introspection.get_table_list(cursor)
+                if t.name == table
+            )
+
+    def assert_column_comment_not_exists(self, table, column):
+        with connection.cursor() as cursor:
+            columns = connection.introspection.get_table_description(cursor, table)
+        self.assertFalse(any([c.name == column and c.comment for c in columns]))
+
     def assertIndexOrder(self, table, index, order):
         constraints = self.get_constraints(table)
         self.assertIn(index, constraints)
@@ -4390,6 +4411,186 @@ def test_add_unique_charfield(self):
             ],
         )
 
+    @skipUnlessDBFeature("supports_comments")
+    def test_add_db_comment_charfield(self):
+        comment = "Custom comment"
+        field = CharField(max_length=255, db_comment=comment)
+        field.set_attributes_from_name("name_with_comment")
+        with connection.schema_editor() as editor:
+            editor.create_model(Author)
+            editor.add_field(Author, field)
+        self.assertEqual(
+            self.get_column_comment(Author._meta.db_table, "name_with_comment"),
+            comment,
+        )
+
+    @skipUnlessDBFeature("supports_comments")
+    def test_add_db_comment_and_default_charfield(self):
+        comment = "Custom comment with default"
+        field = CharField(max_length=255, default="Joe Doe", db_comment=comment)
+        field.set_attributes_from_name("name_with_comment_default")
+        with connection.schema_editor() as editor:
+            editor.create_model(Author)
+            Author.objects.create(name="Before adding a new field")
+            editor.add_field(Author, field)
+
+        self.assertEqual(
+            self.get_column_comment(Author._meta.db_table, "name_with_comment_default"),
+            comment,
+        )
+        with connection.cursor() as cursor:
+            cursor.execute(
+                f"SELECT name_with_comment_default FROM {Author._meta.db_table};"
+            )
+            for row in cursor.fetchall():
+                self.assertEqual(row[0], "Joe Doe")
+
+    @skipUnlessDBFeature("supports_comments")
+    def test_alter_db_comment(self):
+        with connection.schema_editor() as editor:
+            editor.create_model(Author)
+        # Add comment.
+        old_field = Author._meta.get_field("name")
+        new_field = CharField(max_length=255, db_comment="Custom comment")
+        new_field.set_attributes_from_name("name")
+        with connection.schema_editor() as editor:
+            editor.alter_field(Author, old_field, new_field, strict=True)
+        self.assertEqual(
+            self.get_column_comment(Author._meta.db_table, "name"),
+            "Custom comment",
+        )
+        # Alter comment.
+        old_field = new_field
+        new_field = CharField(max_length=255, db_comment="New custom comment")
+        new_field.set_attributes_from_name("name")
+        with connection.schema_editor() as editor:
+            editor.alter_field(Author, old_field, new_field, strict=True)
+        self.assertEqual(
+            self.get_column_comment(Author._meta.db_table, "name"),
+            "New custom comment",
+        )
+        # Remove comment.
+        old_field = new_field
+        new_field = CharField(max_length=255)
+        new_field.set_attributes_from_name("name")
+        with connection.schema_editor() as editor:
+            editor.alter_field(Author, old_field, new_field, strict=True)
+        self.assertIn(
+            self.get_column_comment(Author._meta.db_table, "name"),
+            [None, ""],
+        )
+
+    @skipUnlessDBFeature("supports_comments", "supports_foreign_keys")
+    def test_alter_db_comment_foreign_key(self):
+        with connection.schema_editor() as editor:
+            editor.create_model(Author)
+            editor.create_model(Book)
+
+        comment = "FK custom comment"
+        old_field = Book._meta.get_field("author")
+        new_field = ForeignKey(Author, CASCADE, db_comment=comment)
+        new_field.set_attributes_from_name("author")
+        with connection.schema_editor() as editor:
+            editor.alter_field(Book, old_field, new_field, strict=True)
+        self.assertEqual(
+            self.get_column_comment(Book._meta.db_table, "author_id"),
+            comment,
+        )
+
+    @skipUnlessDBFeature("supports_comments")
+    def test_alter_field_type_preserve_comment(self):
+        with connection.schema_editor() as editor:
+            editor.create_model(Author)
+
+        comment = "This is the name."
+        old_field = Author._meta.get_field("name")
+        new_field = CharField(max_length=255, db_comment=comment)
+        new_field.set_attributes_from_name("name")
+        new_field.model = Author
+        with connection.schema_editor() as editor:
+            editor.alter_field(Author, old_field, new_field, strict=True)
+        self.assertEqual(
+            self.get_column_comment(Author._meta.db_table, "name"),
+            comment,
+        )
+        # Changing a field type should preserve the comment.
+        old_field = new_field
+        new_field = CharField(max_length=511, db_comment=comment)
+        new_field.set_attributes_from_name("name")
+        new_field.model = Author
+        with connection.schema_editor() as editor:
+            editor.alter_field(Author, new_field, old_field, strict=True)
+        # Comment is preserved.
+        self.assertEqual(
+            self.get_column_comment(Author._meta.db_table, "name"),
+            comment,
+        )
+
+    @isolate_apps("schema")
+    @skipUnlessDBFeature("supports_comments")
+    def test_db_comment_table(self):
+        class ModelWithDbTableComment(Model):
+            class Meta:
+                app_label = "schema"
+                db_table_comment = "Custom table comment"
+
+        with connection.schema_editor() as editor:
+            editor.create_model(ModelWithDbTableComment)
+        self.isolated_local_models = [ModelWithDbTableComment]
+        self.assertEqual(
+            self.get_table_comment(ModelWithDbTableComment._meta.db_table),
+            "Custom table comment",
+        )
+        # Alter table comment.
+        old_db_table_comment = ModelWithDbTableComment._meta.db_table_comment
+        with connection.schema_editor() as editor:
+            editor.alter_db_table_comment(
+                ModelWithDbTableComment, old_db_table_comment, "New table comment"
+            )
+        self.assertEqual(
+            self.get_table_comment(ModelWithDbTableComment._meta.db_table),
+            "New table comment",
+        )
+        # Remove table comment.
+        old_db_table_comment = ModelWithDbTableComment._meta.db_table_comment
+        with connection.schema_editor() as editor:
+            editor.alter_db_table_comment(
+                ModelWithDbTableComment, old_db_table_comment, None
+            )
+        self.assertIn(
+            self.get_table_comment(ModelWithDbTableComment._meta.db_table),
+            [None, ""],
+        )
+
+    @isolate_apps("schema")
+    @skipUnlessDBFeature("supports_comments", "supports_foreign_keys")
+    def test_db_comments_from_abstract_model(self):
+        class AbstractModelWithDbComments(Model):
+            name = CharField(
+                max_length=255, db_comment="Custom comment", null=True, blank=True
+            )
+
+            class Meta:
+                app_label = "schema"
+                abstract = True
+                db_table_comment = "Custom table comment"
+
+        class ModelWithDbComments(AbstractModelWithDbComments):
+            pass
+
+        with connection.schema_editor() as editor:
+            editor.create_model(ModelWithDbComments)
+        self.isolated_local_models = [ModelWithDbComments]
+
+        self.assertEqual(
+            self.get_column_comment(ModelWithDbComments._meta.db_table, "name"),
+            "Custom comment",
+        )
+        self.assertEqual(
+            self.get_table_comment(ModelWithDbComments._meta.db_table),
+            "Custom table comment",
+        )
+
     @unittest.skipUnless(connection.vendor == "postgresql", "PostgreSQL specific")
     def test_alter_field_add_index_to_charfield(self):
         # Create the table and verify no initial indexes.

```


## Code snippets

### 1 - django/db/backends/mysql/schema.py:

Start line: 1, End line: 42

```python
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    sql_rename_table = "RENAME TABLE %(old_table)s TO %(new_table)s"

    sql_alter_column_null = "MODIFY %(column)s %(type)s NULL"
    sql_alter_column_not_null = "MODIFY %(column)s %(type)s NOT NULL"
    sql_alter_column_type = "MODIFY %(column)s %(type)s%(collation)s"
    sql_alter_column_no_default_null = "ALTER COLUMN %(column)s SET DEFAULT NULL"

    # No 'CASCADE' which works as a no-op in MySQL but is undocumented
    sql_delete_column = "ALTER TABLE %(table)s DROP COLUMN %(column)s"

    sql_delete_unique = "ALTER TABLE %(table)s DROP INDEX %(name)s"
    sql_create_column_inline_fk = (
        ", ADD CONSTRAINT %(name)s FOREIGN KEY (%(column)s) "
        "REFERENCES %(to_table)s(%(to_column)s)"
    )
    sql_delete_fk = "ALTER TABLE %(table)s DROP FOREIGN KEY %(name)s"

    sql_delete_index = "DROP INDEX %(name)s ON %(table)s"
    sql_rename_index = "ALTER TABLE %(table)s RENAME INDEX %(old_name)s TO %(new_name)s"

    sql_create_pk = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s PRIMARY KEY (%(columns)s)"
    )
    sql_delete_pk = "ALTER TABLE %(table)s DROP PRIMARY KEY"

    sql_create_index = "CREATE INDEX %(name)s ON %(table)s (%(columns)s)%(extra)s"

    @property
    def sql_delete_check(self):
        if self.connection.mysql_is_mariadb:
            # The name of the column check constraint is the same as the field
            # name on MariaDB. Adding IF EXISTS clause prevents migrations
            # crash. Constraint is removed during a "MODIFY" column statement.
            return "ALTER TABLE %(table)s DROP CONSTRAINT IF EXISTS %(name)s"
        return "ALTER TABLE %(table)s DROP CHECK %(name)s"
```
### 2 - django/db/backends/sqlite3/schema.py:

Start line: 100, End line: 121

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_db_table(
        self, model, old_db_table, new_db_table, disable_constraints=True
    ):
        if (
            not self.connection.features.supports_atomic_references_rename
            and disable_constraints
            and self._is_referenced_by_fk_constraint(old_db_table)
        ):
            if self.connection.in_atomic_block:
                raise NotSupportedError(
                    (
                        "Renaming the %r table while in a transaction is not "
                        "supported on SQLite < 3.26 because it would break referential "
                        "integrity. Try adding `atomic = False` to the Migration class."
                    )
                    % old_db_table
                )
            self.connection.enable_constraint_checking()
            super().alter_db_table(model, old_db_table, new_db_table)
            self.connection.disable_constraint_checking()
        else:
            super().alter_db_table(model, old_db_table, new_db_table)
```
### 3 - django/db/backends/mysql/schema.py:

Start line: 104, End line: 118

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def add_field(self, model, field):
        super().add_field(model, field)

        # Simulate the effect of a one-off default.
        # field.default may be unhashable, so a set isn't used for "in" check.
        if self.skip_default(field) and field.default not in (None, NOT_PROVIDED):
            effective_default = self.effective_default(field)
            self.execute(
                "UPDATE %(table)s SET %(column)s = %%s"
                % {
                    "table": self.quote_name(model._meta.db_table),
                    "column": self.quote_name(field.column),
                },
                [effective_default],
            )
```
### 4 - django/db/backends/postgresql/schema.py:

Start line: 143, End line: 256

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    # Setting all constraints to IMMEDIATE to allow changing data in the same
    sql_update_with_default =
    # ... other code

    def _alter_column_type_sql(
        self, model, old_field, new_field, new_type, old_collation, new_collation
    ):
        # Drop indexes on varchar/text/citext columns that are changing to a
        # different type.
        old_db_params = old_field.db_parameters(connection=self.connection)
        old_type = old_db_params["type"]
        if (old_field.db_index or old_field.unique) and (
            (old_type.startswith("varchar") and not new_type.startswith("varchar"))
            or (old_type.startswith("text") and not new_type.startswith("text"))
            or (old_type.startswith("citext") and not new_type.startswith("citext"))
        ):
            index_name = self._create_index_name(
                model._meta.db_table, [old_field.column], suffix="_like"
            )
            self.execute(self._delete_index_sql(model, index_name))

        self.sql_alter_column_type = (
            "ALTER COLUMN %(column)s TYPE %(type)s%(collation)s"
        )
        # Cast when data type changed.
        if using_sql := self._using_sql(new_field, old_field):
            self.sql_alter_column_type += using_sql
        new_internal_type = new_field.get_internal_type()
        old_internal_type = old_field.get_internal_type()
        # Make ALTER TYPE with IDENTITY make sense.
        table = strip_quotes(model._meta.db_table)
        auto_field_types = {
            "AutoField",
            "BigAutoField",
            "SmallAutoField",
        }
        old_is_auto = old_internal_type in auto_field_types
        new_is_auto = new_internal_type in auto_field_types
        if new_is_auto and not old_is_auto:
            column = strip_quotes(new_field.column)
            return (
                (
                    self.sql_alter_column_type
                    % {
                        "column": self.quote_name(column),
                        "type": new_type,
                        "collation": "",
                    },
                    [],
                ),
                [
                    (
                        self.sql_add_identity
                        % {
                            "table": self.quote_name(table),
                            "column": self.quote_name(column),
                        },
                        [],
                    ),
                ],
            )
        elif old_is_auto and not new_is_auto:
            # Drop IDENTITY if exists (pre-Django 4.1 serial columns don't have
            # it).
            self.execute(
                self.sql_drop_indentity
                % {
                    "table": self.quote_name(table),
                    "column": self.quote_name(strip_quotes(new_field.column)),
                }
            )
            column = strip_quotes(new_field.column)
            fragment, _ = super()._alter_column_type_sql(
                model, old_field, new_field, new_type, old_collation, new_collation
            )
            # Drop the sequence if exists (Django 4.1+ identity columns don't
            # have it).
            other_actions = []
            if sequence_name := self._get_sequence_name(table, column):
                other_actions = [
                    (
                        self.sql_delete_sequence
                        % {
                            "sequence": self.quote_name(sequence_name),
                        },
                        [],
                    )
                ]
            return fragment, other_actions
        elif new_is_auto and old_is_auto and old_internal_type != new_internal_type:
            fragment, _ = super()._alter_column_type_sql(
                model, old_field, new_field, new_type, old_collation, new_collation
            )
            column = strip_quotes(new_field.column)
            db_types = {
                "AutoField": "integer",
                "BigAutoField": "bigint",
                "SmallAutoField": "smallint",
            }
            # Alter the sequence type if exists (Django 4.1+ identity columns
            # don't have it).
            other_actions = []
            if sequence_name := self._get_sequence_name(table, column):
                other_actions = [
                    (
                        self.sql_alter_sequence_type
                        % {
                            "sequence": self.quote_name(sequence_name),
                            "type": db_types[new_internal_type],
                        },
                        [],
                    ),
                ]
            return fragment, other_actions
        else:
            return super()._alter_column_type_sql(
                model, old_field, new_field, new_type, old_collation, new_collation
            )
    # ... other code
```
### 5 - django/db/backends/oracle/schema.py:

Start line: 1, End line: 28

```python
import copy
import datetime
import re

from django.db import DatabaseError
from django.db.backends.base.schema import (
    BaseDatabaseSchemaEditor,
    _related_non_m2m_objects,
)
from django.utils.duration import duration_iso_string


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    sql_create_column = "ALTER TABLE %(table)s ADD %(column)s %(definition)s"
    sql_alter_column_type = "MODIFY %(column)s %(type)s%(collation)s"
    sql_alter_column_null = "MODIFY %(column)s NULL"
    sql_alter_column_not_null = "MODIFY %(column)s NOT NULL"
    sql_alter_column_default = "MODIFY %(column)s DEFAULT %(default)s"
    sql_alter_column_no_default = "MODIFY %(column)s DEFAULT NULL"
    sql_alter_column_no_default_null = sql_alter_column_no_default

    sql_delete_column = "ALTER TABLE %(table)s DROP COLUMN %(column)s"
    sql_create_column_inline_fk = (
        "CONSTRAINT %(name)s REFERENCES %(to_table)s(%(to_column)s)%(deferrable)s"
    )
    sql_delete_table = "DROP TABLE %(table)s CASCADE CONSTRAINTS"
    sql_create_index = "CREATE INDEX %(name)s ON %(table)s (%(columns)s)%(extra)s"
```
### 6 - django/db/backends/postgresql/schema.py:

Start line: 1, End line: 84

```python
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import IndexColumns
from django.db.backends.postgresql.psycopg_any import sql
from django.db.backends.utils import strip_quotes


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    # Setting all constraints to IMMEDIATE to allow changing data in the same
    # transaction.
    sql_update_with_default = (
        "UPDATE %(table)s SET %(column)s = %(default)s WHERE %(column)s IS NULL"
        "; SET CONSTRAINTS ALL IMMEDIATE"
    )
    sql_alter_sequence_type = "ALTER SEQUENCE IF EXISTS %(sequence)s AS %(type)s"
    sql_delete_sequence = "DROP SEQUENCE IF EXISTS %(sequence)s CASCADE"

    sql_create_index = (
        "CREATE INDEX %(name)s ON %(table)s%(using)s "
        "(%(columns)s)%(include)s%(extra)s%(condition)s"
    )
    sql_create_index_concurrently = (
        "CREATE INDEX CONCURRENTLY %(name)s ON %(table)s%(using)s "
        "(%(columns)s)%(include)s%(extra)s%(condition)s"
    )
    sql_delete_index = "DROP INDEX IF EXISTS %(name)s"
    sql_delete_index_concurrently = "DROP INDEX CONCURRENTLY IF EXISTS %(name)s"

    # Setting the constraint to IMMEDIATE to allow changing data in the same
    # transaction.
    sql_create_column_inline_fk = (
        "CONSTRAINT %(name)s REFERENCES %(to_table)s(%(to_column)s)%(deferrable)s"
        "; SET CONSTRAINTS %(namespace)s%(name)s IMMEDIATE"
    )
    # Setting the constraint to IMMEDIATE runs any deferred checks to allow
    # dropping it in the same transaction.
    sql_delete_fk = (
        "SET CONSTRAINTS %(name)s IMMEDIATE; "
        "ALTER TABLE %(table)s DROP CONSTRAINT %(name)s"
    )
    sql_delete_procedure = "DROP FUNCTION %(procedure)s(%(param_types)s)"

    def execute(self, sql, params=()):
        # Merge the query client-side, as PostgreSQL won't do it server-side.
        if params is None:
            return super().execute(sql, params)
        sql = self.connection.ops.compose_sql(str(sql), params)
        # Don't let the superclass touch anything.
        return super().execute(sql, None)

    sql_add_identity = (
        "ALTER TABLE %(table)s ALTER COLUMN %(column)s ADD "
        "GENERATED BY DEFAULT AS IDENTITY"
    )
    sql_drop_indentity = (
        "ALTER TABLE %(table)s ALTER COLUMN %(column)s DROP IDENTITY IF EXISTS"
    )

    def quote_value(self, value):
        if isinstance(value, str):
            value = value.replace("%", "%%")
        return sql.quote(value, self.connection.connection)

    def _field_indexes_sql(self, model, field):
        output = super()._field_indexes_sql(model, field)
        like_index_statement = self._create_like_index_sql(model, field)
        if like_index_statement is not None:
            output.append(like_index_statement)
        return output

    def _field_data_type(self, field):
        if field.is_relation:
            return field.rel_db_type(self.connection)
        return self.connection.data_types.get(
            field.get_internal_type(),
            field.db_type(self.connection),
        )

    def _field_base_data_types(self, field):
        # Yield base data types for array fields.
        if field.base_field.get_internal_type() == "ArrayField":
            yield from self._field_base_data_types(field.base_field)
        else:
            yield self._field_data_type(field.base_field)
    # ... other code
```
### 7 - django/db/backends/postgresql/schema.py:

Start line: 313, End line: 338

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    # Setting all constraints to IMMEDIATE to allow changing data in the same
    sql_update_with_default =
    # ... other code

    def _index_columns(self, table, columns, col_suffixes, opclasses):
        if opclasses:
            return IndexColumns(
                table,
                columns,
                self.quote_name,
                col_suffixes=col_suffixes,
                opclasses=opclasses,
            )
        return super()._index_columns(table, columns, col_suffixes, opclasses)

    def add_index(self, model, index, concurrently=False):
        self.execute(
            index.create_sql(model, self, concurrently=concurrently), params=None
        )

    def remove_index(self, model, index, concurrently=False):
        self.execute(index.remove_sql(model, self, concurrently=concurrently))

    def _delete_index_sql(self, model, name, sql=None, concurrently=False):
        sql = (
            self.sql_delete_index_concurrently
            if concurrently
            else self.sql_delete_index
        )
        return super()._delete_index_sql(model, name, sql)
    # ... other code
```
### 8 - django/db/backends/base/schema.py:

Start line: 75, End line: 151

```python
class BaseDatabaseSchemaEditor:
    """
    This class and its subclasses are responsible for emitting schema-changing
    statements to the databases - model creation/removal/alteration, field
    renaming, index fiddling, and so on.
    """

    # Overrideable SQL templates
    sql_create_table = "CREATE TABLE %(table)s (%(definition)s)"
    sql_rename_table = "ALTER TABLE %(old_table)s RENAME TO %(new_table)s"
    sql_retablespace_table = "ALTER TABLE %(table)s SET TABLESPACE %(new_tablespace)s"
    sql_delete_table = "DROP TABLE %(table)s CASCADE"

    sql_create_column = "ALTER TABLE %(table)s ADD COLUMN %(column)s %(definition)s"
    sql_alter_column = "ALTER TABLE %(table)s %(changes)s"
    sql_alter_column_type = "ALTER COLUMN %(column)s TYPE %(type)s%(collation)s"
    sql_alter_column_null = "ALTER COLUMN %(column)s DROP NOT NULL"
    sql_alter_column_not_null = "ALTER COLUMN %(column)s SET NOT NULL"
    sql_alter_column_default = "ALTER COLUMN %(column)s SET DEFAULT %(default)s"
    sql_alter_column_no_default = "ALTER COLUMN %(column)s DROP DEFAULT"
    sql_alter_column_no_default_null = sql_alter_column_no_default
    sql_delete_column = "ALTER TABLE %(table)s DROP COLUMN %(column)s CASCADE"
    sql_rename_column = (
        "ALTER TABLE %(table)s RENAME COLUMN %(old_column)s TO %(new_column)s"
    )
    sql_update_with_default = (
        "UPDATE %(table)s SET %(column)s = %(default)s WHERE %(column)s IS NULL"
    )

    sql_unique_constraint = "UNIQUE (%(columns)s)%(deferrable)s"
    sql_check_constraint = "CHECK (%(check)s)"
    sql_delete_constraint = "ALTER TABLE %(table)s DROP CONSTRAINT %(name)s"
    sql_constraint = "CONSTRAINT %(name)s %(constraint)s"

    sql_create_check = "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s CHECK (%(check)s)"
    sql_delete_check = sql_delete_constraint

    sql_create_unique = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s "
        "UNIQUE (%(columns)s)%(deferrable)s"
    )
    sql_delete_unique = sql_delete_constraint

    sql_create_fk = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s FOREIGN KEY (%(column)s) "
        "REFERENCES %(to_table)s (%(to_column)s)%(deferrable)s"
    )
    sql_create_inline_fk = None
    sql_create_column_inline_fk = None
    sql_delete_fk = sql_delete_constraint

    sql_create_index = (
        "CREATE INDEX %(name)s ON %(table)s "
        "(%(columns)s)%(include)s%(extra)s%(condition)s"
    )
    sql_create_unique_index = (
        "CREATE UNIQUE INDEX %(name)s ON %(table)s "
        "(%(columns)s)%(include)s%(condition)s"
    )
    sql_rename_index = "ALTER INDEX %(old_name)s RENAME TO %(new_name)s"
    sql_delete_index = "DROP INDEX %(name)s"

    sql_create_pk = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s PRIMARY KEY (%(columns)s)"
    )
    sql_delete_pk = sql_delete_constraint

    sql_delete_procedure = "DROP PROCEDURE %(procedure)s"

    def __init__(self, connection, collect_sql=False, atomic=True):
        self.connection = connection
        self.collect_sql = collect_sql
        if self.collect_sql:
            self.collected_sql = []
        self.atomic_migration = self.connection.features.can_rollback_ddl and atomic

    # State-managing methods
```
### 9 - django/db/backends/postgresql/schema.py:

Start line: 277, End line: 311

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    # Setting all constraints to IMMEDIATE to allow changing data in the same
    sql_update_with_default =
    # ... other code

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        super()._alter_field(
            model,
            old_field,
            new_field,
            old_type,
            new_type,
            old_db_params,
            new_db_params,
            strict,
        )
        # Added an index? Create any PostgreSQL-specific indexes.
        if (not (old_field.db_index or old_field.unique) and new_field.db_index) or (
            not old_field.unique and new_field.unique
        ):
            like_index_statement = self._create_like_index_sql(model, new_field)
            if like_index_statement is not None:
                self.execute(like_index_statement)

        # Removed an index? Drop any PostgreSQL-specific indexes.
        if old_field.unique and not (new_field.db_index or new_field.unique):
            index_to_remove = self._create_index_name(
                model._meta.db_table, [old_field.column], suffix="_like"
            )
            self.execute(self._delete_index_sql(model, index_to_remove))
    # ... other code
```
### 10 - django/db/backends/postgresql/schema.py:

Start line: 258, End line: 275

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    # Setting all constraints to IMMEDIATE to allow changing data in the same
    sql_update_with_default =
    # ... other code

    def _alter_column_collation_sql(
        self, model, new_field, new_type, new_collation, old_field
    ):
        sql = self.sql_alter_column_collate
        # Cast when data type changed.
        if using_sql := self._using_sql(new_field, old_field):
            sql += using_sql
        return (
            sql
            % {
                "column": self.quote_name(new_field.column),
                "type": new_type,
                "collation": " " + self._collate_sql(new_collation)
                if new_collation
                else "",
            },
            [],
        )
    # ... other code
```
### 11 - django/db/backends/mysql/schema.py:

Start line: 44, End line: 53

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    @property
    def sql_rename_column(self):
        # MariaDB >= 10.5.2 and MySQL >= 8.0.4 support an
        # "ALTER TABLE ... RENAME COLUMN" statement.
        if self.connection.mysql_is_mariadb:
            if self.connection.mysql_version >= (10, 5, 2):
                return super().sql_rename_column
        elif self.connection.mysql_version >= (8, 0, 4):
            return super().sql_rename_column
        return "ALTER TABLE %(table)s CHANGE %(old_column)s %(new_column)s %(type)s"
```
### 12 - django/db/backends/mysql/schema.py:

Start line: 55, End line: 102

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def quote_value(self, value):
        self.connection.ensure_connection()
        if isinstance(value, str):
            value = value.replace("%", "%%")
        # MySQLdb escapes to string, PyMySQL to bytes.
        quoted = self.connection.connection.escape(
            value, self.connection.connection.encoders
        )
        if isinstance(value, str) and isinstance(quoted, bytes):
            quoted = quoted.decode()
        return quoted

    def _is_limited_data_type(self, field):
        db_type = field.db_type(self.connection)
        return (
            db_type is not None
            and db_type.lower() in self.connection._limited_data_types
        )

    def skip_default(self, field):
        if not self._supports_limited_data_type_defaults:
            return self._is_limited_data_type(field)
        return False

    def skip_default_on_alter(self, field):
        if self._is_limited_data_type(field) and not self.connection.mysql_is_mariadb:
            # MySQL doesn't support defaults for BLOB and TEXT in the
            # ALTER COLUMN statement.
            return True
        return False

    @property
    def _supports_limited_data_type_defaults(self):
        # MariaDB and MySQL >= 8.0.13 support defaults for BLOB and TEXT.
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 13)

    def _column_default_sql(self, field):
        if (
            not self.connection.mysql_is_mariadb
            and self._supports_limited_data_type_defaults
            and self._is_limited_data_type(field)
        ):
            # MySQL supports defaults for BLOB and TEXT columns only if the
            # default value is written as an expression i.e. in parentheses.
            return "(%s)"
        return super()._column_default_sql(field)
```
### 14 - django/db/backends/base/schema.py:

Start line: 1639, End line: 1663

```python
class BaseDatabaseSchemaEditor:

    def _check_sql(self, name, check):
        return self.sql_constraint % {
            "name": self.quote_name(name),
            "constraint": self.sql_check_constraint % {"check": check},
        }

    def _create_check_sql(self, model, name, check):
        return Statement(
            self.sql_create_check,
            table=Table(model._meta.db_table, self.quote_name),
            name=self.quote_name(name),
            check=check,
        )

    def _delete_check_sql(self, model, name):
        if not self.connection.features.supports_table_check_constraints:
            return None
        return self._delete_constraint_sql(self.sql_delete_check, model, name)

    def _delete_constraint_sql(self, template, model, name):
        return Statement(
            template,
            table=Table(model._meta.db_table, self.quote_name),
            name=self.quote_name(name),
        )
```
### 18 - django/db/backends/base/schema.py:

Start line: 889, End line: 970

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        # ... other code
        if (
            old_field.db_index
            and not old_field.unique
            and (not new_field.db_index or new_field.unique)
        ):
            # Find the index for this field
            meta_index_names = {index.name for index in model._meta.indexes}
            # Retrieve only BTREE indexes since this is what's created with
            # db_index=True.
            index_names = self._constraint_names(
                model,
                [old_field.column],
                index=True,
                type_=Index.suffix,
                exclude=meta_index_names,
            )
            for index_name in index_names:
                # The only way to check if an index was created with
                # db_index=True or with Index(['field'], name='foo')
                # is to look at its name (refs #28053).
                self.execute(self._delete_index_sql(model, index_name))
        # Change check constraints?
        if old_db_params["check"] != new_db_params["check"] and old_db_params["check"]:
            meta_constraint_names = {
                constraint.name for constraint in model._meta.constraints
            }
            constraint_names = self._constraint_names(
                model,
                [old_field.column],
                check=True,
                exclude=meta_constraint_names,
            )
            if strict and len(constraint_names) != 1:
                raise ValueError(
                    "Found wrong number (%s) of check constraints for %s.%s"
                    % (
                        len(constraint_names),
                        model._meta.db_table,
                        old_field.column,
                    )
                )
            for constraint_name in constraint_names:
                self.execute(self._delete_check_sql(model, constraint_name))
        # Have they renamed the column?
        if old_field.column != new_field.column:
            self.execute(
                self._rename_field_sql(
                    model._meta.db_table, old_field, new_field, new_type
                )
            )
            # Rename all references to the renamed column.
            for sql in self.deferred_sql:
                if isinstance(sql, Statement):
                    sql.rename_column_references(
                        model._meta.db_table, old_field.column, new_field.column
                    )
        # Next, start accumulating actions to do
        actions = []
        null_actions = []
        post_actions = []
        # Type suffix change? (e.g. auto increment).
        old_type_suffix = old_field.db_type_suffix(connection=self.connection)
        new_type_suffix = new_field.db_type_suffix(connection=self.connection)
        # Type or collation change?
        if (
            old_type != new_type
            or old_type_suffix != new_type_suffix
            or old_collation != new_collation
        ):
            fragment, other_actions = self._alter_column_type_sql(
                model, old_field, new_field, new_type, old_collation, new_collation
            )
            actions.append(fragment)
            post_actions.extend(other_actions)
        # When changing a column NULL constraint to NOT NULL with a given
        # default value, we need to perform 4 steps:
        #  1. Add a default for new incoming writes
        #  2. Update existing NULL rows with new default
        #  3. Replace NULL constraint with NOT NULL
        #  4. Drop the default again.
        # Default change?
        needs_database_default = False
        # ... other code
```
### 21 - django/db/backends/base/schema.py:

Start line: 971, End line: 1058

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        # ... other code
        if old_field.null and not new_field.null:
            old_default = self.effective_default(old_field)
            new_default = self.effective_default(new_field)
            if (
                not self.skip_default_on_alter(new_field)
                and old_default != new_default
                and new_default is not None
            ):
                needs_database_default = True
                actions.append(
                    self._alter_column_default_sql(model, old_field, new_field)
                )
        # Nullability change?
        if old_field.null != new_field.null:
            fragment = self._alter_column_null_sql(model, old_field, new_field)
            if fragment:
                null_actions.append(fragment)
        # Only if we have a default and there is a change from NULL to NOT NULL
        four_way_default_alteration = new_field.has_default() and (
            old_field.null and not new_field.null
        )
        if actions or null_actions:
            if not four_way_default_alteration:
                # If we don't have to do a 4-way default alteration we can
                # directly run a (NOT) NULL alteration
                actions += null_actions
            # Combine actions together if we can (e.g. postgres)
            if self.connection.features.supports_combined_alters and actions:
                sql, params = tuple(zip(*actions))
                actions = [(", ".join(sql), sum(params, []))]
            # Apply those actions
            for sql, params in actions:
                self.execute(
                    self.sql_alter_column
                    % {
                        "table": self.quote_name(model._meta.db_table),
                        "changes": sql,
                    },
                    params,
                )
            if four_way_default_alteration:
                # Update existing rows with default value
                self.execute(
                    self.sql_update_with_default
                    % {
                        "table": self.quote_name(model._meta.db_table),
                        "column": self.quote_name(new_field.column),
                        "default": "%s",
                    },
                    [new_default],
                )
                # Since we didn't run a NOT NULL change before we need to do it
                # now
                for sql, params in null_actions:
                    self.execute(
                        self.sql_alter_column
                        % {
                            "table": self.quote_name(model._meta.db_table),
                            "changes": sql,
                        },
                        params,
                    )
        if post_actions:
            for sql, params in post_actions:
                self.execute(sql, params)
        # If primary_key changed to False, delete the primary key constraint.
        if old_field.primary_key and not new_field.primary_key:
            self._delete_primary_key(model, strict)
        # Added a unique?
        if self._unique_should_be_added(old_field, new_field):
            self.execute(self._create_unique_sql(model, [new_field]))
        # Added an index? Add an index if db_index switched to True or a unique
        # constraint will no longer be used in lieu of an index. The following
        # lines from the truth table show all True cases; the rest are False:
        #
        # old_field.db_index | old_field.unique | new_field.db_index | new_field.unique
        # ------------------------------------------------------------------------------
        # False              | False            | True               | False
        # False              | True             | True               | False
        # True               | True             | True               | False
        if (
            (not old_field.db_index or old_field.unique)
            and new_field.db_index
            and not new_field.unique
        ):
            self.execute(self._create_index_sql(model, fields=[new_field]))
        # Type alteration on primary key? Then we need to alter the column
        # referring to us.
        # ... other code
```
### 22 - django/db/backends/mysql/features.py:

Start line: 147, End line: 254

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def _mysql_storage_engine(self):
        "Internal method used in Django tests. Don't rely on this from your code"
        return self.connection.mysql_server_data["default_storage_engine"]

    @cached_property
    def allows_auto_pk_0(self):
        """
        Autoincrement primary key can be set to 0 if it doesn't generate new
        autoincrement values.
        """
        return "NO_AUTO_VALUE_ON_ZERO" in self.connection.sql_mode

    @cached_property
    def update_can_self_select(self):
        return self.connection.mysql_is_mariadb and self.connection.mysql_version >= (
            10,
            3,
            2,
        )

    @cached_property
    def can_introspect_foreign_keys(self):
        "Confirm support for introspected foreign keys"
        return self._mysql_storage_engine != "MyISAM"

    @cached_property
    def introspected_field_types(self):
        return {
            **super().introspected_field_types,
            "BinaryField": "TextField",
            "BooleanField": "IntegerField",
            "DurationField": "BigIntegerField",
            "GenericIPAddressField": "CharField",
        }

    @cached_property
    def can_return_columns_from_insert(self):
        return self.connection.mysql_is_mariadb and self.connection.mysql_version >= (
            10,
            5,
            0,
        )

    can_return_rows_from_bulk_insert = property(
        operator.attrgetter("can_return_columns_from_insert")
    )

    @cached_property
    def has_zoneinfo_database(self):
        return self.connection.mysql_server_data["has_zoneinfo_database"]

    @cached_property
    def is_sql_auto_is_null_enabled(self):
        return self.connection.mysql_server_data["sql_auto_is_null"]

    @cached_property
    def supports_over_clause(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 2)

    supports_frame_range_fixed_distance = property(
        operator.attrgetter("supports_over_clause")
    )

    @cached_property
    def supports_column_check_constraints(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 16)

    supports_table_check_constraints = property(
        operator.attrgetter("supports_column_check_constraints")
    )

    @cached_property
    def can_introspect_check_constraints(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 16)

    @cached_property
    def has_select_for_update_skip_locked(self):
        if self.connection.mysql_is_mariadb:
            return self.connection.mysql_version >= (10, 6)
        return self.connection.mysql_version >= (8, 0, 1)

    @cached_property
    def has_select_for_update_nowait(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 1)

    @cached_property
    def has_select_for_update_of(self):
        return (
            not self.connection.mysql_is_mariadb
            and self.connection.mysql_version >= (8, 0, 1)
        )

    @cached_property
    def supports_explain_analyze(self):
        return self.connection.mysql_is_mariadb or self.connection.mysql_version >= (
            8,
            0,
            18,
        )
```
### 23 - django/db/backends/base/schema.py:

Start line: 1547, End line: 1598

```python
class BaseDatabaseSchemaEditor:

    def _create_unique_sql(
        self,
        model,
        fields,
        name=None,
        condition=None,
        deferrable=None,
        include=None,
        opclasses=None,
        expressions=None,
    ):
        if (
            (
                deferrable
                and not self.connection.features.supports_deferrable_unique_constraints
            )
            or (condition and not self.connection.features.supports_partial_indexes)
            or (include and not self.connection.features.supports_covering_indexes)
            or (
                expressions and not self.connection.features.supports_expression_indexes
            )
        ):
            return None

        compiler = Query(model, alias_cols=False).get_compiler(
            connection=self.connection
        )
        table = model._meta.db_table
        columns = [field.column for field in fields]
        if name is None:
            name = self._unique_constraint_name(table, columns, quote=True)
        else:
            name = self.quote_name(name)
        if condition or include or opclasses or expressions:
            sql = self.sql_create_unique_index
        else:
            sql = self.sql_create_unique
        if columns:
            columns = self._index_columns(
                table, columns, col_suffixes=(), opclasses=opclasses
            )
        else:
            columns = Expressions(table, expressions, compiler, self.quote_value)
        return Statement(
            sql,
            table=Table(table, self.quote_name),
            name=name,
            columns=columns,
            condition=self._index_condition_sql(condition),
            deferrable=self._deferrable_constraint_sql(deferrable),
            include=self._index_include_sql(model, include),
        )
```
### 24 - django/db/backends/base/schema.py:

Start line: 1059, End line: 1133

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        # ... other code
        rels_to_update = []
        if drop_foreign_keys:
            rels_to_update.extend(_related_non_m2m_objects(old_field, new_field))
        # Changed to become primary key?
        if self._field_became_primary_key(old_field, new_field):
            # Make the new one
            self.execute(self._create_primary_key_sql(model, new_field))
            # Update all referencing columns
            rels_to_update.extend(_related_non_m2m_objects(old_field, new_field))
        # Handle our type alters on the other end of rels from the PK stuff above
        for old_rel, new_rel in rels_to_update:
            rel_db_params = new_rel.field.db_parameters(connection=self.connection)
            rel_type = rel_db_params["type"]
            rel_collation = rel_db_params.get("collation")
            old_rel_db_params = old_rel.field.db_parameters(connection=self.connection)
            old_rel_collation = old_rel_db_params.get("collation")
            fragment, other_actions = self._alter_column_type_sql(
                new_rel.related_model,
                old_rel.field,
                new_rel.field,
                rel_type,
                old_rel_collation,
                rel_collation,
            )
            self.execute(
                self.sql_alter_column
                % {
                    "table": self.quote_name(new_rel.related_model._meta.db_table),
                    "changes": fragment[0],
                },
                fragment[1],
            )
            for sql, params in other_actions:
                self.execute(sql, params)
        # Does it have a foreign key?
        if (
            self.connection.features.supports_foreign_keys
            and new_field.remote_field
            and (
                fks_dropped or not old_field.remote_field or not old_field.db_constraint
            )
            and new_field.db_constraint
        ):
            self.execute(
                self._create_fk_sql(model, new_field, "_fk_%(to_table)s_%(to_column)s")
            )
        # Rebuild FKs that pointed to us if we previously had to drop them
        if drop_foreign_keys:
            for _, rel in rels_to_update:
                if rel.field.db_constraint:
                    self.execute(
                        self._create_fk_sql(rel.related_model, rel.field, "_fk")
                    )
        # Does it have check constraints we need to add?
        if old_db_params["check"] != new_db_params["check"] and new_db_params["check"]:
            constraint_name = self._create_index_name(
                model._meta.db_table, [new_field.column], suffix="_check"
            )
            self.execute(
                self._create_check_sql(model, constraint_name, new_db_params["check"])
            )
        # Drop the default if we need to
        # (Django usually does not use in-database defaults)
        if needs_database_default:
            changes_sql, params = self._alter_column_default_sql(
                model, old_field, new_field, drop=True
            )
            sql = self.sql_alter_column % {
                "table": self.quote_name(model._meta.db_table),
                "changes": changes_sql,
            }
            self.execute(sql, params)
        # Reset connection if required
        if self.connection.features.connection_persists_old_columns:
            self.connection.close()
```
### 30 - django/db/backends/base/features.py:

Start line: 368, End line: 386

```python
class BaseDatabaseFeatures:
    minimum_database_version =
    # ... other code

    @cached_property
    def supports_transactions(self):
        """Confirm support for transactions."""
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE TABLE ROLLBACK_TEST (X INT)")
            self.connection.set_autocommit(False)
            cursor.execute("INSERT INTO ROLLBACK_TEST (X) VALUES (8)")
            self.connection.rollback()
            self.connection.set_autocommit(True)
            cursor.execute("SELECT COUNT(X) FROM ROLLBACK_TEST")
            (count,) = cursor.fetchone()
            cursor.execute("DROP TABLE ROLLBACK_TEST")
        return count == 0

    def allows_group_by_selected_pks_on_model(self, model):
        if not self.allows_group_by_selected_pks:
            return False
        return model._meta.managed
```
### 31 - django/db/backends/base/schema.py:

Start line: 284, End line: 339

```python
class BaseDatabaseSchemaEditor:

    # Field <-> database mapping functions

    def _iter_column_sql(
        self, column_db_type, params, model, field, field_db_params, include_default
    ):
        yield column_db_type
        if collation := field_db_params.get("collation"):
            yield self._collate_sql(collation)
        # Work out nullability.
        null = field.null
        # Include a default value, if requested.
        include_default = (
            include_default
            and not self.skip_default(field)
            and
            # Don't include a default value if it's a nullable field and the
            # default cannot be dropped in the ALTER COLUMN statement (e.g.
            # MySQL longtext and longblob).
            not (null and self.skip_default_on_alter(field))
        )
        if include_default:
            default_value = self.effective_default(field)
            if default_value is not None:
                column_default = "DEFAULT " + self._column_default_sql(field)
                if self.connection.features.requires_literal_defaults:
                    # Some databases can't take defaults as a parameter (Oracle).
                    # If this is the case, the individual schema backend should
                    # implement prepare_default().
                    yield column_default % self.prepare_default(default_value)
                else:
                    yield column_default
                    params.append(default_value)
        # Oracle treats the empty string ('') as null, so coerce the null
        # option whenever '' is a possible value.
        if (
            field.empty_strings_allowed
            and not field.primary_key
            and self.connection.features.interprets_empty_strings_as_nulls
        ):
            null = True
        if not null:
            yield "NOT NULL"
        elif not self.connection.features.implied_column_null:
            yield "NULL"
        if field.primary_key:
            yield "PRIMARY KEY"
        elif field.unique:
            yield "UNIQUE"
        # Optionally add the tablespace if it's an implicitly indexed column.
        tablespace = field.db_tablespace or model._meta.db_tablespace
        if (
            tablespace
            and self.connection.features.supports_tablespaces
            and field.unique
        ):
            yield self.connection.ops.tablespace_sql(tablespace, inline=True)
```
### 33 - django/db/backends/base/schema.py:

Start line: 1376, End line: 1392

```python
class BaseDatabaseSchemaEditor:

    def _delete_index_sql(self, model, name, sql=None):
        return Statement(
            sql or self.sql_delete_index,
            table=Table(model._meta.db_table, self.quote_name),
            name=self.quote_name(name),
        )

    def _rename_index_sql(self, model, old_name, new_name):
        return Statement(
            self.sql_rename_index,
            table=Table(model._meta.db_table, self.quote_name),
            old_name=self.quote_name(old_name),
            new_name=self.quote_name(new_name),
        )

    def _index_columns(self, table, columns, col_suffixes, opclasses):
        return Columns(table, columns, self.quote_name, col_suffixes=col_suffixes)
```
### 34 - django/db/backends/base/schema.py:

Start line: 153, End line: 201

```python
class BaseDatabaseSchemaEditor:

    def __enter__(self):
        self.deferred_sql = []
        if self.atomic_migration:
            self.atomic = atomic(self.connection.alias)
            self.atomic.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            for sql in self.deferred_sql:
                self.execute(sql)
        if self.atomic_migration:
            self.atomic.__exit__(exc_type, exc_value, traceback)

    # Core utility functions

    def execute(self, sql, params=()):
        """Execute the given SQL statement, with optional parameters."""
        # Don't perform the transactional DDL check if SQL is being collected
        # as it's not going to be executed anyway.
        if (
            not self.collect_sql
            and self.connection.in_atomic_block
            and not self.connection.features.can_rollback_ddl
        ):
            raise TransactionManagementError(
                "Executing DDL statements while in a transaction on databases "
                "that can't perform a rollback is prohibited."
            )
        # Account for non-string statement objects.
        sql = str(sql)
        # Log the command we're running, then run it
        logger.debug(
            "%s; (params %r)", sql, params, extra={"params": params, "sql": sql}
        )
        if self.collect_sql:
            ending = "" if sql.rstrip().endswith(";") else ";"
            if params is not None:
                self.collected_sql.append(
                    (sql % tuple(map(self.quote_value, params))) + ending
                )
            else:
                self.collected_sql.append(sql + ending)
        else:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)

    def quote_name(self, name):
        return self.connection.ops.quote_name(name)
```
### 35 - django/db/backends/base/schema.py:

Start line: 1441, End line: 1460

```python
class BaseDatabaseSchemaEditor:

    def _field_should_be_indexed(self, model, field):
        return field.db_index and not field.unique

    def _field_became_primary_key(self, old_field, new_field):
        return not old_field.primary_key and new_field.primary_key

    def _unique_should_be_added(self, old_field, new_field):
        return (
            not new_field.primary_key
            and new_field.unique
            and (not old_field.unique or old_field.primary_key)
        )

    def _rename_field_sql(self, table, old_field, new_field, new_type):
        return self.sql_rename_column % {
            "table": self.quote_name(table),
            "old_column": self.quote_name(old_field.column),
            "new_column": self.quote_name(new_field.column),
            "type": new_type,
        }
```
### 36 - django/db/backends/base/schema.py:

Start line: 598, End line: 626

```python
class BaseDatabaseSchemaEditor:

    def alter_db_table(self, model, old_db_table, new_db_table):
        """Rename the table a model points to."""
        if old_db_table == new_db_table or (
            self.connection.features.ignores_table_name_case
            and old_db_table.lower() == new_db_table.lower()
        ):
            return
        self.execute(
            self.sql_rename_table
            % {
                "old_table": self.quote_name(old_db_table),
                "new_table": self.quote_name(new_db_table),
            }
        )
        # Rename all references to the old table name.
        for sql in self.deferred_sql:
            if isinstance(sql, Statement):
                sql.rename_table_references(old_db_table, new_db_table)

    def alter_db_tablespace(self, model, old_db_tablespace, new_db_tablespace):
        """Move a model's table between tablespaces."""
        self.execute(
            self.sql_retablespace_table
            % {
                "table": self.quote_name(model._meta.db_table),
                "old_tablespace": self.quote_name(old_db_tablespace),
                "new_tablespace": self.quote_name(new_db_tablespace),
            }
        )
```
### 37 - django/db/backends/base/schema.py:

Start line: 1462, End line: 1481

```python
class BaseDatabaseSchemaEditor:

    def _create_fk_sql(self, model, field, suffix):
        table = Table(model._meta.db_table, self.quote_name)
        name = self._fk_constraint_name(model, field, suffix)
        column = Columns(model._meta.db_table, [field.column], self.quote_name)
        to_table = Table(field.target_field.model._meta.db_table, self.quote_name)
        to_column = Columns(
            field.target_field.model._meta.db_table,
            [field.target_field.column],
            self.quote_name,
        )
        deferrable = self.connection.ops.deferrable_sql()
        return Statement(
            self.sql_create_fk,
            table=table,
            name=name,
            column=column,
            to_table=to_table,
            to_column=to_column,
            deferrable=deferrable,
        )
```
### 38 - django/db/backends/base/schema.py:

Start line: 628, End line: 700

```python
class BaseDatabaseSchemaEditor:

    def add_field(self, model, field):
        """
        Create a field on a model. Usually involves adding a column, but may
        involve adding a table instead (for M2M fields).
        """
        # Special-case implicit M2M tables
        if field.many_to_many and field.remote_field.through._meta.auto_created:
            return self.create_model(field.remote_field.through)
        # Get the column's definition
        definition, params = self.column_sql(model, field, include_default=True)
        # It might not actually have a column behind it
        if definition is None:
            return
        if col_type_suffix := field.db_type_suffix(connection=self.connection):
            definition += f" {col_type_suffix}"
        # Check constraints can go on the column SQL here
        db_params = field.db_parameters(connection=self.connection)
        if db_params["check"]:
            definition += " " + self.sql_check_constraint % db_params
        if (
            field.remote_field
            and self.connection.features.supports_foreign_keys
            and field.db_constraint
        ):
            constraint_suffix = "_fk_%(to_table)s_%(to_column)s"
            # Add FK constraint inline, if supported.
            if self.sql_create_column_inline_fk:
                to_table = field.remote_field.model._meta.db_table
                to_column = field.remote_field.model._meta.get_field(
                    field.remote_field.field_name
                ).column
                namespace, _ = split_identifier(model._meta.db_table)
                definition += " " + self.sql_create_column_inline_fk % {
                    "name": self._fk_constraint_name(model, field, constraint_suffix),
                    "namespace": "%s." % self.quote_name(namespace)
                    if namespace
                    else "",
                    "column": self.quote_name(field.column),
                    "to_table": self.quote_name(to_table),
                    "to_column": self.quote_name(to_column),
                    "deferrable": self.connection.ops.deferrable_sql(),
                }
            # Otherwise, add FK constraints later.
            else:
                self.deferred_sql.append(
                    self._create_fk_sql(model, field, constraint_suffix)
                )
        # Build the SQL and run it
        sql = self.sql_create_column % {
            "table": self.quote_name(model._meta.db_table),
            "column": self.quote_name(field.column),
            "definition": definition,
        }
        self.execute(sql, params)
        # Drop the default if we need to
        # (Django usually does not use in-database defaults)
        if (
            not self.skip_default_on_alter(field)
            and self.effective_default(field) is not None
        ):
            changes_sql, params = self._alter_column_default_sql(
                model, None, field, drop=True
            )
            sql = self.sql_alter_column % {
                "table": self.quote_name(model._meta.db_table),
                "changes": changes_sql,
            }
            self.execute(sql, params)
        # Add an index, if required
        self.deferred_sql.extend(self._field_indexes_sql(model, field))
        # Reset connection if required
        if self.connection.features.connection_persists_old_columns:
            self.connection.close()
```
### 43 - django/db/models/base.py:

Start line: 1, End line: 66

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

from asgiref.sync import sync_to_async

import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
    NON_FIELD_ERRORS,
    FieldDoesNotExist,
    FieldError,
    MultipleObjectsReturned,
    ObjectDoesNotExist,
    ValidationError,
)
from django.db import (
    DJANGO_VERSION_PICKLE_KEY,
    DatabaseError,
    connection,
    connections,
    router,
    transaction,
)
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import RawSQL
from django.db.models.fields.related import (
    ForeignObjectRel,
    OneToOneField,
    lazy_related_operation,
    resolve_relation,
)
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
    class_prepared,
    post_init,
    post_save,
    pre_init,
    pre_save,
)
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _


class Deferred:
    def __repr__(self):
        return "<Deferred field>"

    def __str__(self):
        return "<Deferred field>"


DEFERRED = Deferred()
```
### 44 - django/db/backends/base/schema.py:

Start line: 438, End line: 455

```python
class BaseDatabaseSchemaEditor:

    def create_model(self, model):
        """
        Create a table and any accompanying indexes or unique constraints for
        the given `model`.
        """
        sql, params = self.table_sql(model)
        # Prevent using [] as params, in the case a literal '%' is used in the
        # definition.
        self.execute(sql, params or None)

        # Add any field index and index_together's (deferred as SQLite
        # _remake_table needs it).
        self.deferred_sql.extend(self._model_indexes_sql(model))

        # Make M2M tables
        for field in model._meta.local_many_to_many:
            if field.remote_field.through._meta.auto_created:
                self.create_model(field.remote_field.through)
```
### 45 - django/db/backends/base/schema.py:

Start line: 478, End line: 520

```python
class BaseDatabaseSchemaEditor:

    def add_index(self, model, index):
        """Add an index on a model."""
        if (
            index.contains_expressions
            and not self.connection.features.supports_expression_indexes
        ):
            return None
        # Index.create_sql returns interpolated SQL which makes params=None a
        # necessity to avoid escaping attempts on execution.
        self.execute(index.create_sql(model, self), params=None)

    def remove_index(self, model, index):
        """Remove an index from a model."""
        if (
            index.contains_expressions
            and not self.connection.features.supports_expression_indexes
        ):
            return None
        self.execute(index.remove_sql(model, self))

    def rename_index(self, model, old_index, new_index):
        if self.connection.features.can_rename_index:
            self.execute(
                self._rename_index_sql(model, old_index.name, new_index.name),
                params=None,
            )
        else:
            self.remove_index(model, old_index)
            self.add_index(model, new_index)

    def add_constraint(self, model, constraint):
        """Add a constraint to a model."""
        sql = constraint.create_sql(model, self)
        if sql:
            # Constraint.create_sql returns interpolated SQL which makes
            # params=None a necessity to avoid escaping attempts on execution.
            self.execute(sql, params=None)

    def remove_constraint(self, model, constraint):
        """Remove a constraint from a model."""
        sql = constraint.remove_sql(model, self)
        if sql:
            self.execute(sql)
```
### 46 - django/db/backends/mysql/introspection.py:

Start line: 214, End line: 225

```python
class DatabaseIntrospection(BaseDatabaseIntrospection):

    def _parse_constraint_columns(self, check_clause, columns):
        check_columns = OrderedSet()
        statement = sqlparse.parse(check_clause)[0]
        tokens = (token for token in statement.flatten() if not token.is_whitespace)
        for token in tokens:
            if (
                token.ttype == sqlparse.tokens.Name
                and self.connection.ops.quote_name(token.value) == token.value
                and token.value[1:-1] in columns
            ):
                check_columns.add(token.value[1:-1])
        return check_columns
```
### 47 - django/core/management/commands/inspectdb.py:

Start line: 54, End line: 253

```python
class Command(BaseCommand):

    def handle_inspection(self, options):
        connection = connections[options["database"]]
        # 'table_name_filter' is a stealth option
        table_name_filter = options.get("table_name_filter")

        def table2model(table_name):
            return re.sub(r"[^a-zA-Z0-9]", "", table_name.title())

        with connection.cursor() as cursor:
            yield "# This is an auto-generated Django model module."
            yield "# You'll have to do the following manually to clean this up:"
            yield "#   * Rearrange models' order"
            yield "#   * Make sure each model has one field with primary_key=True"
            yield (
                "#   * Make sure each ForeignKey and OneToOneField has `on_delete` set "
                "to the desired behavior"
            )
            yield (
                "#   * Remove `managed = False` lines if you wish to allow "
                "Django to create, modify, and delete the table"
            )
            yield (
                "# Feel free to rename the models, but don't rename db_table values or "
                "field names."
            )
            yield "from %s import models" % self.db_module
            known_models = []
            table_info = connection.introspection.get_table_list(cursor)

            # Determine types of tables and/or views to be introspected.
            types = {"t"}
            if options["include_partitions"]:
                types.add("p")
            if options["include_views"]:
                types.add("v")

            for table_name in options["table"] or sorted(
                info.name for info in table_info if info.type in types
            ):
                if table_name_filter is not None and callable(table_name_filter):
                    if not table_name_filter(table_name):
                        continue
                try:
                    try:
                        relations = connection.introspection.get_relations(
                            cursor, table_name
                        )
                    except NotImplementedError:
                        relations = {}
                    try:
                        constraints = connection.introspection.get_constraints(
                            cursor, table_name
                        )
                    except NotImplementedError:
                        constraints = {}
                    primary_key_columns = (
                        connection.introspection.get_primary_key_columns(
                            cursor, table_name
                        )
                    )
                    primary_key_column = (
                        primary_key_columns[0] if primary_key_columns else None
                    )
                    unique_columns = [
                        c["columns"][0]
                        for c in constraints.values()
                        if c["unique"] and len(c["columns"]) == 1
                    ]
                    table_description = connection.introspection.get_table_description(
                        cursor, table_name
                    )
                except Exception as e:
                    yield "# Unable to inspect table '%s'" % table_name
                    yield "# The error was: %s" % e
                    continue

                model_name = table2model(table_name)
                yield ""
                yield ""
                yield "class %s(models.Model):" % model_name
                known_models.append(model_name)
                used_column_names = []  # Holds column names used in the table so far
                column_to_field_name = {}  # Maps column names to names of model fields
                used_relations = set()  # Holds foreign relations used in the table.
                for row in table_description:
                    comment_notes = (
                        []
                    )  # Holds Field notes, to be displayed in a Python comment.
                    extra_params = {}  # Holds Field parameters such as 'db_column'.
                    column_name = row.name
                    is_relation = column_name in relations

                    att_name, params, notes = self.normalize_col_name(
                        column_name, used_column_names, is_relation
                    )
                    extra_params.update(params)
                    comment_notes.extend(notes)

                    used_column_names.append(att_name)
                    column_to_field_name[column_name] = att_name

                    # Add primary_key and unique, if necessary.
                    if column_name == primary_key_column:
                        extra_params["primary_key"] = True
                        if len(primary_key_columns) > 1:
                            comment_notes.append(
                                "The composite primary key (%s) found, that is not "
                                "supported. The first column is selected."
                                % ", ".join(primary_key_columns)
                            )
                    elif column_name in unique_columns:
                        extra_params["unique"] = True

                    if is_relation:
                        ref_db_column, ref_db_table = relations[column_name]
                        if extra_params.pop("unique", False) or extra_params.get(
                            "primary_key"
                        ):
                            rel_type = "OneToOneField"
                        else:
                            rel_type = "ForeignKey"
                            ref_pk_column = (
                                connection.introspection.get_primary_key_column(
                                    cursor, ref_db_table
                                )
                            )
                            if ref_pk_column and ref_pk_column != ref_db_column:
                                extra_params["to_field"] = ref_db_column
                        rel_to = (
                            "self"
                            if ref_db_table == table_name
                            else table2model(ref_db_table)
                        )
                        if rel_to in known_models:
                            field_type = "%s(%s" % (rel_type, rel_to)
                        else:
                            field_type = "%s('%s'" % (rel_type, rel_to)
                        if rel_to in used_relations:
                            extra_params["related_name"] = "%s_%s_set" % (
                                model_name.lower(),
                                att_name,
                            )
                        used_relations.add(rel_to)
                    else:
                        # Calling `get_field_type` to get the field type string and any
                        # additional parameters and notes.
                        field_type, field_params, field_notes = self.get_field_type(
                            connection, table_name, row
                        )
                        extra_params.update(field_params)
                        comment_notes.extend(field_notes)

                        field_type += "("

                    # Don't output 'id = meta.AutoField(primary_key=True)', because
                    # that's assumed if it doesn't exist.
                    if att_name == "id" and extra_params == {"primary_key": True}:
                        if field_type == "AutoField(":
                            continue
                        elif (
                            field_type
                            == connection.features.introspected_field_types["AutoField"]
                            + "("
                        ):
                            comment_notes.append("AutoField?")

                    # Add 'null' and 'blank', if the 'null_ok' flag was present in the
                    # table description.
                    if row.null_ok:  # If it's NULL...
                        extra_params["blank"] = True
                        extra_params["null"] = True

                    field_desc = "%s = %s%s" % (
                        att_name,
                        # Custom fields will have a dotted path
                        "" if "." in field_type else "models.",
                        field_type,
                    )
                    if field_type.startswith(("ForeignKey(", "OneToOneField(")):
                        field_desc += ", models.DO_NOTHING"

                    if extra_params:
                        if not field_desc.endswith("("):
                            field_desc += ", "
                        field_desc += ", ".join(
                            "%s=%r" % (k, v) for k, v in extra_params.items()
                        )
                    field_desc += ")"
                    if comment_notes:
                        field_desc += "  # " + " ".join(comment_notes)
                    yield "    %s" % field_desc
                is_view = any(
                    info.name == table_name and info.type == "v" for info in table_info
                )
                is_partition = any(
                    info.name == table_name and info.type == "p" for info in table_info
                )
                yield from self.get_meta(
                    table_name, constraints, column_to_field_name, is_view, is_partition
                )
```
### 49 - django/db/backends/base/schema.py:

Start line: 1600, End line: 1637

```python
class BaseDatabaseSchemaEditor:

    def _unique_constraint_name(self, table, columns, quote=True):
        if quote:

            def create_unique_name(*args, **kwargs):
                return self.quote_name(self._create_index_name(*args, **kwargs))

        else:
            create_unique_name = self._create_index_name

        return IndexName(table, columns, "_uniq", create_unique_name)

    def _delete_unique_sql(
        self,
        model,
        name,
        condition=None,
        deferrable=None,
        include=None,
        opclasses=None,
        expressions=None,
    ):
        if (
            (
                deferrable
                and not self.connection.features.supports_deferrable_unique_constraints
            )
            or (condition and not self.connection.features.supports_partial_indexes)
            or (include and not self.connection.features.supports_covering_indexes)
            or (
                expressions and not self.connection.features.supports_expression_indexes
            )
        ):
            return None
        if condition or include or opclasses or expressions:
            sql = self.sql_delete_index
        else:
            sql = self.sql_delete_unique
        return self._delete_constraint_sql(sql, model, name)
```
### 50 - django/db/backends/base/schema.py:

Start line: 1483, End line: 1505

```python
class BaseDatabaseSchemaEditor:

    def _fk_constraint_name(self, model, field, suffix):
        def create_fk_name(*args, **kwargs):
            return self.quote_name(self._create_index_name(*args, **kwargs))

        return ForeignKeyName(
            model._meta.db_table,
            [field.column],
            split_identifier(field.target_field.model._meta.db_table)[1],
            [field.target_field.column],
            suffix,
            create_fk_name,
        )

    def _delete_fk_sql(self, model, name):
        return self._delete_constraint_sql(self.sql_delete_fk, model, name)

    def _deferrable_constraint_sql(self, deferrable):
        if deferrable is None:
            return ""
        if deferrable == Deferrable.DEFERRED:
            return " DEFERRABLE INITIALLY DEFERRED"
        if deferrable == Deferrable.IMMEDIATE:
            return " DEFERRABLE INITIALLY IMMEDIATE"
```
### 51 - django/db/backends/mysql/features.py:

Start line: 256, End line: 324

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def supported_explain_formats(self):
        # Alias MySQL's TRADITIONAL to TEXT for consistency with other
        # backends.
        formats = {"JSON", "TEXT", "TRADITIONAL"}
        if not self.connection.mysql_is_mariadb and self.connection.mysql_version >= (
            8,
            0,
            16,
        ):
            formats.add("TREE")
        return formats

    @cached_property
    def supports_transactions(self):
        """
        All storage engines except MyISAM support transactions.
        """
        return self._mysql_storage_engine != "MyISAM"

    uses_savepoints = property(operator.attrgetter("supports_transactions"))
    can_release_savepoints = property(operator.attrgetter("supports_transactions"))

    @cached_property
    def ignores_table_name_case(self):
        return self.connection.mysql_server_data["lower_case_table_names"]

    @cached_property
    def supports_default_in_lead_lag(self):
        # To be added in https://jira.mariadb.org/browse/MDEV-12981.
        return not self.connection.mysql_is_mariadb

    @cached_property
    def can_introspect_json_field(self):
        if self.connection.mysql_is_mariadb:
            return self.can_introspect_check_constraints
        return True

    @cached_property
    def supports_index_column_ordering(self):
        if self._mysql_storage_engine != "InnoDB":
            return False
        if self.connection.mysql_is_mariadb:
            return self.connection.mysql_version >= (10, 8)
        return self.connection.mysql_version >= (8, 0, 1)

    @cached_property
    def supports_expression_indexes(self):
        return (
            not self.connection.mysql_is_mariadb
            and self._mysql_storage_engine != "MyISAM"
            and self.connection.mysql_version >= (8, 0, 13)
        )

    @cached_property
    def supports_select_intersection(self):
        is_mariadb = self.connection.mysql_is_mariadb
        return is_mariadb or self.connection.mysql_version >= (8, 0, 31)

    supports_select_difference = property(
        operator.attrgetter("supports_select_intersection")
    )

    @cached_property
    def can_rename_index(self):
        if self.connection.mysql_is_mariadb:
            return self.connection.mysql_version >= (10, 5, 2)
        return True
```
### 54 - django/db/models/options.py:

Start line: 173, End line: 242

```python
class Options:

    def contribute_to_class(self, cls, name):
        from django.db import connection
        from django.db.backends.utils import truncate_name

        cls._meta = self
        self.model = cls
        # First, construct the default values for these options.
        self.object_name = cls.__name__
        self.model_name = self.object_name.lower()
        self.verbose_name = camel_case_to_spaces(self.object_name)

        # Store the original user-defined values for each option,
        # for use when serializing the model definition
        self.original_attrs = {}

        # Next, apply any overridden values from 'class Meta'.
        if self.meta:
            meta_attrs = self.meta.__dict__.copy()
            for name in self.meta.__dict__:
                # Ignore any private attributes that Django doesn't care about.
                # NOTE: We can't modify a dictionary's contents while looping
                # over it, so we loop over the *original* dictionary instead.
                if name.startswith("_"):
                    del meta_attrs[name]
            for attr_name in DEFAULT_NAMES:
                if attr_name in meta_attrs:
                    setattr(self, attr_name, meta_attrs.pop(attr_name))
                    self.original_attrs[attr_name] = getattr(self, attr_name)
                elif hasattr(self.meta, attr_name):
                    setattr(self, attr_name, getattr(self.meta, attr_name))
                    self.original_attrs[attr_name] = getattr(self, attr_name)

            self.unique_together = normalize_together(self.unique_together)
            self.index_together = normalize_together(self.index_together)
            if self.index_together:
                warnings.warn(
                    f"'index_together' is deprecated. Use 'Meta.indexes' in "
                    f"{self.label!r} instead.",
                    RemovedInDjango51Warning,
                )
            # App label/class name interpolation for names of constraints and
            # indexes.
            if not getattr(cls._meta, "abstract", False):
                for attr_name in {"constraints", "indexes"}:
                    objs = getattr(self, attr_name, [])
                    setattr(self, attr_name, self._format_names_with_class(cls, objs))

            # verbose_name_plural is a special case because it uses a 's'
            # by default.
            if self.verbose_name_plural is None:
                self.verbose_name_plural = format_lazy("{}s", self.verbose_name)

            # order_with_respect_and ordering are mutually exclusive.
            self._ordering_clash = bool(self.ordering and self.order_with_respect_to)

            # Any leftover attributes must be invalid.
            if meta_attrs != {}:
                raise TypeError(
                    "'class Meta' got invalid attribute(s): %s" % ",".join(meta_attrs)
                )
        else:
            self.verbose_name_plural = format_lazy("{}s", self.verbose_name)
        del self.meta

        # If the db_table wasn't provided, use the app_label + model_name.
        if not self.db_table:
            self.db_table = "%s_%s" % (self.app_label, self.model_name)
            self.db_table = truncate_name(
                self.db_table, connection.ops.max_name_length()
            )
```
### 55 - django/db/backends/mysql/features.py:

Start line: 1, End line: 61

```python
import operator

from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    empty_fetchmany_value = ()
    allows_group_by_selected_pks = True
    related_fields_match_type = True
    # MySQL doesn't support sliced subqueries with IN/ALL/ANY/SOME.
    allow_sliced_subqueries_with_in = False
    has_select_for_update = True
    supports_forward_references = False
    supports_regex_backreferencing = False
    supports_date_lookup_using_string = False
    supports_timezones = False
    requires_explicit_null_ordering_when_grouping = True
    atomic_transactions = False
    can_clone_databases = True
    supports_temporal_subtraction = True
    supports_slicing_ordering_in_compound = True
    supports_index_on_text_field = False
    supports_update_conflicts = True
    create_test_procedure_without_params_sql = """
        CREATE PROCEDURE test_procedure ()
        BEGIN
            DECLARE V_I INTEGER;
            SET V_I = 1;
        END;
    """
    create_test_procedure_with_int_param_sql = """
        CREATE PROCEDURE test_procedure (P_I INTEGER)
        BEGIN
            DECLARE V_I INTEGER;
            SET V_I = P_I;
        END;
    """
    create_test_table_with_composite_primary_key = """
        CREATE TABLE test_table_composite_pk (
            column_1 INTEGER NOT NULL,
            column_2 INTEGER NOT NULL,
            PRIMARY KEY(column_1, column_2)
        )
    """
    # Neither MySQL nor MariaDB support partial indexes.
    supports_partial_indexes = False
    # COLLATE must be wrapped in parentheses because MySQL treats COLLATE as an
    # indexed expression.
    collate_as_index_expression = True

    supports_order_by_nulls_modifier = False
    order_by_nulls_first = True
    supports_logical_xor = True

    @cached_property
    def minimum_database_version(self):
        if self.connection.mysql_is_mariadb:
            return (10, 4)
        else:
            return (8,)
```
### 56 - django/db/backends/postgresql/features.py:

Start line: 1, End line: 112

```python
import operator

from django.db import DataError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    minimum_database_version = (12,)
    allows_group_by_selected_pks = True
    can_return_columns_from_insert = True
    can_return_rows_from_bulk_insert = True
    has_real_datatype = True
    has_native_uuid_field = True
    has_native_duration_field = True
    has_native_json_field = True
    can_defer_constraint_checks = True
    has_select_for_update = True
    has_select_for_update_nowait = True
    has_select_for_update_of = True
    has_select_for_update_skip_locked = True
    has_select_for_no_key_update = True
    can_release_savepoints = True
    supports_tablespaces = True
    supports_transactions = True
    can_introspect_materialized_views = True
    can_distinct_on_fields = True
    can_rollback_ddl = True
    schema_editor_uses_clientside_param_binding = True
    supports_combined_alters = True
    nulls_order_largest = True
    closed_cursor_error_class = InterfaceError
    greatest_least_ignores_nulls = True
    can_clone_databases = True
    supports_temporal_subtraction = True
    supports_slicing_ordering_in_compound = True
    create_test_procedure_without_params_sql = """
        CREATE FUNCTION test_procedure () RETURNS void AS $$
        DECLARE
            V_I INTEGER;
        BEGIN
            V_I := 1;
        END;
    $$ LANGUAGE plpgsql;"""
    create_test_procedure_with_int_param_sql = """
        CREATE FUNCTION test_procedure (P_I INTEGER) RETURNS void AS $$
        DECLARE
            V_I INTEGER;
        BEGIN
            V_I := P_I;
        END;
    $$ LANGUAGE plpgsql;"""
    create_test_table_with_composite_primary_key = """
        CREATE TABLE test_table_composite_pk (
            column_1 INTEGER NOT NULL,
            column_2 INTEGER NOT NULL,
            PRIMARY KEY(column_1, column_2)
        )
    """
    requires_casted_case_in_updates = True
    supports_over_clause = True
    only_supports_unbounded_with_preceding_and_following = True
    supports_aggregate_filter_clause = True
    supported_explain_formats = {"JSON", "TEXT", "XML", "YAML"}
    supports_deferrable_unique_constraints = True
    has_json_operators = True
    json_key_contains_list_matching_requires_list = True
    supports_update_conflicts = True
    supports_update_conflicts_with_target = True
    supports_covering_indexes = True
    can_rename_index = True
    test_collations = {
        "non_default": "sv-x-icu",
        "swedish_ci": "sv-x-icu",
    }
    test_now_utc_template = "STATEMENT_TIMESTAMP() AT TIME ZONE 'UTC'"

    django_test_skips = {
        "opclasses are PostgreSQL only.": {
            "indexes.tests.SchemaIndexesNotPostgreSQLTests."
            "test_create_index_ignores_opclasses",
        },
    }

    @cached_property
    def prohibits_null_characters_in_text_exception(self):
        if is_psycopg3:
            return DataError, "PostgreSQL text fields cannot contain NUL (0x00) bytes"
        else:
            return ValueError, "A string literal cannot contain NUL (0x00) characters."

    @cached_property
    def introspected_field_types(self):
        return {
            **super().introspected_field_types,
            "PositiveBigIntegerField": "BigIntegerField",
            "PositiveIntegerField": "IntegerField",
            "PositiveSmallIntegerField": "SmallIntegerField",
        }

    @cached_property
    def is_postgresql_13(self):
        return self.connection.pg_version >= 130000

    @cached_property
    def is_postgresql_14(self):
        return self.connection.pg_version >= 140000

    has_bit_xor = property(operator.attrgetter("is_postgresql_14"))
    supports_covering_spgist_indexes = property(operator.attrgetter("is_postgresql_14"))
```
### 58 - django/db/backends/base/schema.py:

Start line: 734, End line: 796

```python
class BaseDatabaseSchemaEditor:

    def alter_field(self, model, old_field, new_field, strict=False):
        """
        Allow a field's type, uniqueness, nullability, default, column,
        constraints, etc. to be modified.
        `old_field` is required to compute the necessary changes.
        If `strict` is True, raise errors if the old column does not match
        `old_field` precisely.
        """
        if not self._field_should_be_altered(old_field, new_field):
            return
        # Ensure this field is even column-based
        old_db_params = old_field.db_parameters(connection=self.connection)
        old_type = old_db_params["type"]
        new_db_params = new_field.db_parameters(connection=self.connection)
        new_type = new_db_params["type"]
        if (old_type is None and old_field.remote_field is None) or (
            new_type is None and new_field.remote_field is None
        ):
            raise ValueError(
                "Cannot alter field %s into %s - they do not properly define "
                "db_type (are you using a badly-written custom field?)"
                % (old_field, new_field),
            )
        elif (
            old_type is None
            and new_type is None
            and (
                old_field.remote_field.through
                and new_field.remote_field.through
                and old_field.remote_field.through._meta.auto_created
                and new_field.remote_field.through._meta.auto_created
            )
        ):
            return self._alter_many_to_many(model, old_field, new_field, strict)
        elif (
            old_type is None
            and new_type is None
            and (
                old_field.remote_field.through
                and new_field.remote_field.through
                and not old_field.remote_field.through._meta.auto_created
                and not new_field.remote_field.through._meta.auto_created
            )
        ):
            # Both sides have through models; this is a no-op.
            return
        elif old_type is None or new_type is None:
            raise ValueError(
                "Cannot alter field %s into %s - they are not compatible types "
                "(you cannot alter to or from M2M fields, or add or remove "
                "through= on M2M fields)" % (old_field, new_field)
            )

        self._alter_field(
            model,
            old_field,
            new_field,
            old_type,
            new_type,
            old_db_params,
            new_db_params,
            strict,
        )
```
### 59 - django/db/backends/postgresql/introspection.py:

Start line: 79, End line: 124

```python
class DatabaseIntrospection(BaseDatabaseIntrospection):
    data_types_reverse =
    # ... other code

    def get_table_description(self, cursor, table_name):
        """
        Return a description of the table with the DB-API cursor.description
        interface.
        """
        # Query the pg_catalog tables as cursor.description does not reliably
        # return the nullable property and information_schema.columns does not
        # contain details of materialized views.
        cursor.execute(
            """
            SELECT
                a.attname AS column_name,
                NOT (a.attnotnull OR (t.typtype = 'd' AND t.typnotnull)) AS is_nullable,
                pg_get_expr(ad.adbin, ad.adrelid) AS column_default,
                CASE WHEN collname = 'default' THEN NULL ELSE collname END AS collation,
                a.attidentity != '' AS is_autofield
            FROM pg_attribute a
            LEFT JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
            LEFT JOIN pg_collation co ON a.attcollation = co.oid
            JOIN pg_type t ON a.atttypid = t.oid
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE c.relkind IN ('f', 'm', 'p', 'r', 'v')
                AND c.relname = %s
                AND n.nspname NOT IN ('pg_catalog', 'pg_toast')
                AND pg_catalog.pg_table_is_visible(c.oid)
        """,
            [table_name],
        )
        field_map = {line[0]: line[1:] for line in cursor.fetchall()}
        cursor.execute(
            "SELECT * FROM %s LIMIT 1" % self.connection.ops.quote_name(table_name)
        )
        return [
            FieldInfo(
                line.name,
                line.type_code,
                # display_size is always None on psycopg2.
                line.internal_size if line.display_size is None else line.display_size,
                line.internal_size,
                line.precision,
                line.scale,
                *field_map[line.name],
            )
            for line in cursor.description
        ]
    # ... other code
```
### 60 - django/db/backends/mysql/introspection.py:

Start line: 77, End line: 164

```python
class DatabaseIntrospection(BaseDatabaseIntrospection):

    def get_table_description(self, cursor, table_name):
        """
        Return a description of the table with the DB-API cursor.description
        interface."
        """
        json_constraints = {}
        if (
            self.connection.mysql_is_mariadb
            and self.connection.features.can_introspect_json_field
        ):
            # JSON data type is an alias for LONGTEXT in MariaDB, select
            # JSON_VALID() constraints to introspect JSONField.
            cursor.execute(
                """
                SELECT c.constraint_name AS column_name
                FROM information_schema.check_constraints AS c
                WHERE
                    c.table_name = %s AND
                    LOWER(c.check_clause) =
                        'json_valid(`' + LOWER(c.constraint_name) + '`)' AND
                    c.constraint_schema = DATABASE()
                """,
                [table_name],
            )
            json_constraints = {row[0] for row in cursor.fetchall()}
        # A default collation for the given table.
        cursor.execute(
            """
            SELECT  table_collation
            FROM    information_schema.tables
            WHERE   table_schema = DATABASE()
            AND     table_name = %s
            """,
            [table_name],
        )
        row = cursor.fetchone()
        default_column_collation = row[0] if row else ""
        # information_schema database gives more accurate results for some figures:
        # - varchar length returned by cursor.description is an internal length,
        #   not visible length (#5725)
        # - precision and scale (for decimal fields) (#5014)
        # - auto_increment is not available in cursor.description
        cursor.execute(
            """
            SELECT
                column_name, data_type, character_maximum_length,
                numeric_precision, numeric_scale, extra, column_default,
                CASE
                    WHEN collation_name = %s THEN NULL
                    ELSE collation_name
                END AS collation_name,
                CASE
                    WHEN column_type LIKE '%% unsigned' THEN 1
                    ELSE 0
                END AS is_unsigned
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = DATABASE()
            """,
            [default_column_collation, table_name],
        )
        field_info = {line[0]: InfoLine(*line) for line in cursor.fetchall()}

        cursor.execute(
            "SELECT * FROM %s LIMIT 1" % self.connection.ops.quote_name(table_name)
        )

        def to_int(i):
            return int(i) if i is not None else i

        fields = []
        for line in cursor.description:
            info = field_info[line[0]]
            fields.append(
                FieldInfo(
                    *line[:2],
                    to_int(info.max_len) or line[2],
                    to_int(info.max_len) or line[3],
                    to_int(info.num_prec) or line[4],
                    to_int(info.num_scale) or line[5],
                    line[6],
                    info.column_default,
                    info.collation,
                    info.extra,
                    info.is_unsigned,
                    line[0] in json_constraints,
                )
            )
        return fields
```
### 61 - django/db/backends/mysql/features.py:

Start line: 83, End line: 145

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def django_test_skips(self):
        skips = {
            "This doesn't work on MySQL.": {
                "db_functions.comparison.test_greatest.GreatestTests."
                "test_coalesce_workaround",
                "db_functions.comparison.test_least.LeastTests."
                "test_coalesce_workaround",
            },
            "Running on MySQL requires utf8mb4 encoding (#18392).": {
                "model_fields.test_textfield.TextFieldTests.test_emoji",
                "model_fields.test_charfield.TestCharField.test_emoji",
            },
            "MySQL doesn't support functional indexes on a function that "
            "returns JSON": {
                "schema.tests.SchemaTests.test_func_index_json_key_transform",
            },
            "MySQL supports multiplying and dividing DurationFields by a "
            "scalar value but it's not implemented (#25287).": {
                "expressions.tests.FTimeDeltaTests.test_durationfield_multiply_divide",
            },
            "UPDATE ... ORDER BY syntax on MySQL/MariaDB does not support ordering by"
            "related fields.": {
                "update.tests.AdvancedTests."
                "test_update_ordered_by_inline_m2m_annotation",
                "update.tests.AdvancedTests.test_update_ordered_by_m2m_annotation",
            },
        }
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            3,
        ) < self.connection.mysql_version < (10, 5, 2):
            skips.update(
                {
                    "https://jira.mariadb.org/browse/MDEV-19598": {
                        "schema.tests.SchemaTests."
                        "test_alter_not_unique_field_to_primary_key",
                    },
                }
            )
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            12,
        ) < self.connection.mysql_version < (10, 5):
            skips.update(
                {
                    "https://jira.mariadb.org/browse/MDEV-22775": {
                        "schema.tests.SchemaTests."
                        "test_alter_pk_with_self_referential_field",
                    },
                }
            )
        if not self.supports_explain_analyze:
            skips.update(
                {
                    "MariaDB and MySQL >= 8.0.18 specific.": {
                        "queries.test_explain.ExplainTests.test_mysql_analyze",
                    },
                }
            )
        return skips
```
### 63 - django/db/models/base.py:

Start line: 369, End line: 436

```python
class ModelBase(type):

    def add_to_class(cls, name, value):
        if _has_contribute_to_class(value):
            value.contribute_to_class(cls, name)
        else:
            setattr(cls, name, value)

    def _prepare(cls):
        """Create some methods once self._meta has been populated."""
        opts = cls._meta
        opts._prepare(cls)

        if opts.order_with_respect_to:
            cls.get_next_in_order = partialmethod(
                cls._get_next_or_previous_in_order, is_next=True
            )
            cls.get_previous_in_order = partialmethod(
                cls._get_next_or_previous_in_order, is_next=False
            )

            # Defer creating accessors on the foreign class until it has been
            # created and registered. If remote_field is None, we're ordering
            # with respect to a GenericForeignKey and don't know what the
            # foreign class is - we'll add those accessors later in
            # contribute_to_class().
            if opts.order_with_respect_to.remote_field:
                wrt = opts.order_with_respect_to
                remote = wrt.remote_field.model
                lazy_related_operation(make_foreign_order_accessors, cls, remote)

        # Give the class a docstring -- its definition.
        if cls.__doc__ is None:
            cls.__doc__ = "%s(%s)" % (
                cls.__name__,
                ", ".join(f.name for f in opts.fields),
            )

        get_absolute_url_override = settings.ABSOLUTE_URL_OVERRIDES.get(
            opts.label_lower
        )
        if get_absolute_url_override:
            setattr(cls, "get_absolute_url", get_absolute_url_override)

        if not opts.managers:
            if any(f.name == "objects" for f in opts.fields):
                raise ValueError(
                    "Model %s must specify a custom Manager, because it has a "
                    "field named 'objects'." % cls.__name__
                )
            manager = Manager()
            manager.auto_created = True
            cls.add_to_class("objects", manager)

        # Set the name of _meta.indexes. This can't be done in
        # Options.contribute_to_class() because fields haven't been added to
        # the model at that point.
        for index in cls._meta.indexes:
            if not index.name:
                index.set_name_with_model(cls)

        class_prepared.send(sender=cls)

    @property
    def _base_manager(cls):
        return cls._meta.base_manager

    @property
    def _default_manager(cls):
        return cls._meta.default_manager
```
### 65 - django/db/backends/oracle/introspection.py:

Start line: 99, End line: 185

```python
class DatabaseIntrospection(BaseDatabaseIntrospection):

    def get_table_description(self, cursor, table_name):
        """
        Return a description of the table with the DB-API cursor.description
        interface.
        """
        # user_tab_columns gives data default for columns
        cursor.execute(
            """
            SELECT
                user_tab_cols.column_name,
                user_tab_cols.data_default,
                CASE
                    WHEN user_tab_cols.collation = user_tables.default_collation
                    THEN NULL
                    ELSE user_tab_cols.collation
                END collation,
                CASE
                    WHEN user_tab_cols.char_used IS NULL
                    THEN user_tab_cols.data_length
                    ELSE user_tab_cols.char_length
                END as display_size,
                CASE
                    WHEN user_tab_cols.identity_column = 'YES' THEN 1
                    ELSE 0
                END as is_autofield,
                CASE
                    WHEN EXISTS (
                        SELECT  1
                        FROM user_json_columns
                        WHERE
                            user_json_columns.table_name = user_tab_cols.table_name AND
                            user_json_columns.column_name = user_tab_cols.column_name
                    )
                    THEN 1
                    ELSE 0
                END as is_json
            FROM user_tab_cols
            LEFT OUTER JOIN
                user_tables ON user_tables.table_name = user_tab_cols.table_name
            WHERE user_tab_cols.table_name = UPPER(%s)
            """,
            [table_name],
        )
        field_map = {
            column: (
                display_size,
                default if default != "NULL" else None,
                collation,
                is_autofield,
                is_json,
            )
            for (
                column,
                default,
                collation,
                display_size,
                is_autofield,
                is_json,
            ) in cursor.fetchall()
        }
        self.cache_bust_counter += 1
        cursor.execute(
            "SELECT * FROM {} WHERE ROWNUM < 2 AND {} > 0".format(
                self.connection.ops.quote_name(table_name), self.cache_bust_counter
            )
        )
        description = []
        for desc in cursor.description:
            name = desc[0]
            display_size, default, collation, is_autofield, is_json = field_map[name]
            name %= {}  # cx_Oracle, for some reason, doubles percent signs.
            description.append(
                FieldInfo(
                    self.identifier_converter(name),
                    desc[1],
                    display_size,
                    desc[3],
                    desc[4] or 0,
                    desc[5] or 0,
                    *desc[6:],
                    default,
                    collation,
                    is_autofield,
                    is_json,
                )
            )
        return description
```
### 66 - django/db/backends/base/schema.py:

Start line: 1202, End line: 1231

```python
class BaseDatabaseSchemaEditor:

    def _alter_column_type_sql(
        self, model, old_field, new_field, new_type, old_collation, new_collation
    ):
        """
        Hook to specialize column type alteration for different backends,
        for cases when a creation type is different to an alteration type
        (e.g. SERIAL in PostgreSQL, PostGIS fields).

        Return a two-tuple of: an SQL fragment of (sql, params) to insert into
        an ALTER TABLE statement and a list of extra (sql, params) tuples to
        run once the field is altered.
        """
        if collate_sql := self._collate_sql(
            new_collation, old_collation, model._meta.db_table
        ):
            collate_sql = f" {collate_sql}"
        else:
            collate_sql = ""
        return (
            (
                self.sql_alter_column_type
                % {
                    "column": self.quote_name(new_field.column),
                    "type": new_type,
                    "collation": collate_sql,
                },
                [],
            ),
            [],
        )
```
### 67 - django/db/backends/base/schema.py:

Start line: 1706, End line: 1743

```python
class BaseDatabaseSchemaEditor:

    def _delete_primary_key(self, model, strict=False):
        constraint_names = self._constraint_names(model, primary_key=True)
        if strict and len(constraint_names) != 1:
            raise ValueError(
                "Found wrong number (%s) of PK constraints for %s"
                % (
                    len(constraint_names),
                    model._meta.db_table,
                )
            )
        for constraint_name in constraint_names:
            self.execute(self._delete_primary_key_sql(model, constraint_name))

    def _create_primary_key_sql(self, model, field):
        return Statement(
            self.sql_create_pk,
            table=Table(model._meta.db_table, self.quote_name),
            name=self.quote_name(
                self._create_index_name(
                    model._meta.db_table, [field.column], suffix="_pk"
                )
            ),
            columns=Columns(model._meta.db_table, [field.column], self.quote_name),
        )

    def _delete_primary_key_sql(self, model, name):
        return self._delete_constraint_sql(self.sql_delete_pk, model, name)

    def _collate_sql(self, collation, old_collation=None, table_name=None):
        return "COLLATE " + self.quote_name(collation) if collation else ""

    def remove_procedure(self, procedure_name, param_types=()):
        sql = self.sql_delete_procedure % {
            "procedure": self.quote_name(procedure_name),
            "param_types": ",".join(param_types),
        }
        self.execute(sql)
```
### 68 - django/db/backends/base/schema.py:

Start line: 1507, End line: 1545

```python
class BaseDatabaseSchemaEditor:

    def _unique_sql(
        self,
        model,
        fields,
        name,
        condition=None,
        deferrable=None,
        include=None,
        opclasses=None,
        expressions=None,
    ):
        if (
            deferrable
            and not self.connection.features.supports_deferrable_unique_constraints
        ):
            return None
        if condition or include or opclasses or expressions:
            # Databases support conditional, covering, and functional unique
            # constraints via a unique index.
            sql = self._create_unique_sql(
                model,
                fields,
                name=name,
                condition=condition,
                include=include,
                opclasses=opclasses,
                expressions=expressions,
            )
            if sql:
                self.deferred_sql.append(sql)
            return None
        constraint = self.sql_unique_constraint % {
            "columns": ", ".join([self.quote_name(field.column) for field in fields]),
            "deferrable": self._deferrable_constraint_sql(deferrable),
        }
        return self.sql_constraint % {
            "name": self.quote_name(name),
            "constraint": constraint,
        }
```
### 70 - django/db/models/base.py:

Start line: 2160, End line: 2241

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_long_column_names(cls, databases):
        """
        Check that any auto-generated column names are shorter than the limits
        for each database in which the model will be created.
        """
        if not databases:
            return []
        errors = []
        allowed_len = None
        db_alias = None

        # Find the minimum max allowed length among all specified db_aliases.
        for db in databases:
            # skip databases where the model won't be created
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            max_name_length = connection.ops.max_name_length()
            if max_name_length is None or connection.features.truncates_names:
                continue
            else:
                if allowed_len is None:
                    allowed_len = max_name_length
                    db_alias = db
                elif max_name_length < allowed_len:
                    allowed_len = max_name_length
                    db_alias = db

        if allowed_len is None:
            return errors

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Check if auto-generated name for the field is too long
            # for the database.
            if (
                f.db_column is None
                and column_name is not None
                and len(column_name) > allowed_len
            ):
                errors.append(
                    checks.Error(
                        'Autogenerated column name too long for field "%s". '
                        'Maximum length is "%s" for database "%s".'
                        % (column_name, allowed_len, db_alias),
                        hint="Set the column name manually using 'db_column'.",
                        obj=cls,
                        id="models.E018",
                    )
                )

        for f in cls._meta.local_many_to_many:
            # Skip nonexistent models.
            if isinstance(f.remote_field.through, str):
                continue

            # Check if auto-generated name for the M2M field is too long
            # for the database.
            for m2m in f.remote_field.through._meta.local_fields:
                _, rel_name = m2m.get_attname_column()
                if (
                    m2m.db_column is None
                    and rel_name is not None
                    and len(rel_name) > allowed_len
                ):
                    errors.append(
                        checks.Error(
                            "Autogenerated column name too long for M2M field "
                            '"%s". Maximum length is "%s" for database "%s".'
                            % (rel_name, allowed_len, db_alias),
                            hint=(
                                "Use 'through' to create a separate model for "
                                "M2M and then set column_name using 'db_column'."
                            ),
                            obj=cls,
                            id="models.E019",
                        )
                    )

        return errors
```
### 72 - django/db/models/base.py:

Start line: 1526, End line: 1561

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [
            *cls._check_swappable(),
            *cls._check_model(),
            *cls._check_managers(**kwargs),
        ]
        if not cls._meta.swapped:
            databases = kwargs.get("databases") or []
            errors += [
                *cls._check_fields(**kwargs),
                *cls._check_m2m_through_same_relationship(),
                *cls._check_long_column_names(databases),
            ]
            clash_errors = (
                *cls._check_id_field(),
                *cls._check_field_name_clashes(),
                *cls._check_model_name_db_lookup_clashes(),
                *cls._check_property_name_related_field_accessor_clashes(),
                *cls._check_single_primary_key(),
            )
            errors.extend(clash_errors)
            # If there are field name clashes, hide consequent column name
            # clashes.
            if not clash_errors:
                errors.extend(cls._check_column_name_clashes())
            errors += [
                *cls._check_index_together(),
                *cls._check_unique_together(),
                *cls._check_indexes(databases),
                *cls._check_ordering(),
                *cls._check_constraints(databases),
                *cls._check_default_pk(),
            ]

        return errors
```
### 73 - django/db/backends/base/schema.py:

Start line: 1297, End line: 1320

```python
class BaseDatabaseSchemaEditor:

    def _get_index_tablespace_sql(self, model, fields, db_tablespace=None):
        if db_tablespace is None:
            if len(fields) == 1 and fields[0].db_tablespace:
                db_tablespace = fields[0].db_tablespace
            elif settings.DEFAULT_INDEX_TABLESPACE:
                db_tablespace = settings.DEFAULT_INDEX_TABLESPACE
            elif model._meta.db_tablespace:
                db_tablespace = model._meta.db_tablespace
        if db_tablespace is not None:
            return " " + self.connection.ops.tablespace_sql(db_tablespace)
        return ""

    def _index_condition_sql(self, condition):
        if condition:
            return " WHERE " + condition
        return ""

    def _index_include_sql(self, model, columns):
        if not columns or not self.connection.features.supports_covering_indexes:
            return ""
        return Statement(
            " INCLUDE (%(columns)s)",
            columns=Columns(model._meta.db_table, columns, self.quote_name),
        )
```
### 74 - django/db/backends/mysql/schema.py:

Start line: 205, End line: 231

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _delete_composed_index(self, model, fields, *args):
        self._create_missing_fk_index(model, fields=fields)
        return super()._delete_composed_index(model, fields, *args)

    def _set_field_new_type_null_status(self, field, new_type):
        """
        Keep the null property of the old field. If it has changed, it will be
        handled separately.
        """
        if field.null:
            new_type += " NULL"
        else:
            new_type += " NOT NULL"
        return new_type

    def _alter_column_type_sql(
        self, model, old_field, new_field, new_type, old_collation, new_collation
    ):
        new_type = self._set_field_new_type_null_status(old_field, new_type)
        return super()._alter_column_type_sql(
            model, old_field, new_field, new_type, old_collation, new_collation
        )

    def _rename_field_sql(self, table, old_field, new_field, new_type):
        new_type = self._set_field_new_type_null_status(old_field, new_type)
        return super()._rename_field_sql(table, old_field, new_field, new_type)
```
### 77 - django/db/backends/mysql/schema.py:

Start line: 137, End line: 153

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _field_should_be_indexed(self, model, field):
        if not super()._field_should_be_indexed(model, field):
            return False

        storage = self.connection.introspection.get_storage_engine(
            self.connection.cursor(), model._meta.db_table
        )
        # No need to create an index for ForeignKey fields except if
        # db_constraint=False because the index from that constraint won't be
        # created.
        if (
            storage == "InnoDB"
            and field.get_internal_type() == "ForeignKey"
            and field.db_constraint
        ):
            return False
        return not self._is_limited_data_type(field)
```
### 78 - django/db/backends/base/schema.py:

Start line: 1164, End line: 1200

```python
class BaseDatabaseSchemaEditor:

    def _alter_column_default_sql(self, model, old_field, new_field, drop=False):
        """
        Hook to specialize column default alteration.

        Return a (sql, params) fragment to add or drop (depending on the drop
        argument) a default to new_field's column.
        """
        new_default = self.effective_default(new_field)
        default = self._column_default_sql(new_field)
        params = [new_default]

        if drop:
            params = []
        elif self.connection.features.requires_literal_defaults:
            # Some databases (Oracle) can't take defaults as a parameter
            # If this is the case, the SchemaEditor for that database should
            # implement prepare_default().
            default = self.prepare_default(new_default)
            params = []

        new_db_params = new_field.db_parameters(connection=self.connection)
        if drop:
            if new_field.null:
                sql = self.sql_alter_column_no_default_null
            else:
                sql = self.sql_alter_column_no_default
        else:
            sql = self.sql_alter_column_default
        return (
            sql
            % {
                "column": self.quote_name(new_field.column),
                "type": new_db_params["type"],
                "default": default,
            },
            params,
        )
```
### 80 - django/db/backends/base/schema.py:

Start line: 203, End line: 282

```python
class BaseDatabaseSchemaEditor:

    def table_sql(self, model):
        """Take a model and return its table definition."""
        # Add any unique_togethers (always deferred, as some fields might be
        # created afterward, like geometry fields with some backends).
        for field_names in model._meta.unique_together:
            fields = [model._meta.get_field(field) for field in field_names]
            self.deferred_sql.append(self._create_unique_sql(model, fields))
        # Create column SQL, add FK deferreds if needed.
        column_sqls = []
        params = []
        for field in model._meta.local_fields:
            # SQL.
            definition, extra_params = self.column_sql(model, field)
            if definition is None:
                continue
            # Check constraints can go on the column SQL here.
            db_params = field.db_parameters(connection=self.connection)
            if db_params["check"]:
                definition += " " + self.sql_check_constraint % db_params
            # Autoincrement SQL (for backends with inline variant).
            col_type_suffix = field.db_type_suffix(connection=self.connection)
            if col_type_suffix:
                definition += " %s" % col_type_suffix
            params.extend(extra_params)
            # FK.
            if field.remote_field and field.db_constraint:
                to_table = field.remote_field.model._meta.db_table
                to_column = field.remote_field.model._meta.get_field(
                    field.remote_field.field_name
                ).column
                if self.sql_create_inline_fk:
                    definition += " " + self.sql_create_inline_fk % {
                        "to_table": self.quote_name(to_table),
                        "to_column": self.quote_name(to_column),
                    }
                elif self.connection.features.supports_foreign_keys:
                    self.deferred_sql.append(
                        self._create_fk_sql(
                            model, field, "_fk_%(to_table)s_%(to_column)s"
                        )
                    )
            # Add the SQL to our big list.
            column_sqls.append(
                "%s %s"
                % (
                    self.quote_name(field.column),
                    definition,
                )
            )
            # Autoincrement SQL (for backends with post table definition
            # variant).
            if field.get_internal_type() in (
                "AutoField",
                "BigAutoField",
                "SmallAutoField",
            ):
                autoinc_sql = self.connection.ops.autoinc_sql(
                    model._meta.db_table, field.column
                )
                if autoinc_sql:
                    self.deferred_sql.extend(autoinc_sql)
        constraints = [
            constraint.constraint_sql(model, self)
            for constraint in model._meta.constraints
        ]
        sql = self.sql_create_table % {
            "table": self.quote_name(model._meta.db_table),
            "definition": ", ".join(
                str(constraint)
                for constraint in (*column_sqls, *constraints)
                if constraint
            ),
        }
        if model._meta.db_tablespace:
            tablespace_sql = self.connection.ops.tablespace_sql(
                model._meta.db_tablespace
            )
            if tablespace_sql:
                sql += " " + tablespace_sql
        return sql, params
```
### 82 - django/db/models/base.py:

Start line: 938, End line: 1026

```python
class Model(AltersData, metaclass=ModelBase):

    def _save_table(
        self,
        raw=False,
        cls=None,
        force_insert=False,
        force_update=False,
        using=None,
        update_fields=None,
    ):
        """
        Do the heavy-lifting involved in saving. Update or insert the data
        for a single table.
        """
        meta = cls._meta
        non_pks = [f for f in meta.local_concrete_fields if not f.primary_key]

        if update_fields:
            non_pks = [
                f
                for f in non_pks
                if f.name in update_fields or f.attname in update_fields
            ]

        pk_val = self._get_pk_val(meta)
        if pk_val is None:
            pk_val = meta.pk.get_pk_value_on_save(self)
            setattr(self, meta.pk.attname, pk_val)
        pk_set = pk_val is not None
        if not pk_set and (force_update or update_fields):
            raise ValueError("Cannot force an update in save() with no primary key.")
        updated = False
        # Skip an UPDATE when adding an instance and primary key has a default.
        if (
            not raw
            and not force_insert
            and self._state.adding
            and meta.pk.default
            and meta.pk.default is not NOT_PROVIDED
        ):
            force_insert = True
        # If possible, try an UPDATE. If that doesn't update anything, do an INSERT.
        if pk_set and not force_insert:
            base_qs = cls._base_manager.using(using)
            values = [
                (
                    f,
                    None,
                    (getattr(self, f.attname) if raw else f.pre_save(self, False)),
                )
                for f in non_pks
            ]
            forced_update = update_fields or force_update
            updated = self._do_update(
                base_qs, using, pk_val, values, update_fields, forced_update
            )
            if force_update and not updated:
                raise DatabaseError("Forced update did not affect any rows.")
            if update_fields and not updated:
                raise DatabaseError("Save with update_fields did not affect any rows.")
        if not updated:
            if meta.order_with_respect_to:
                # If this is a model with an order_with_respect_to
                # autopopulate the _order field
                field = meta.order_with_respect_to
                filter_args = field.get_filter_kwargs_for_object(self)
                self._order = (
                    cls._base_manager.using(using)
                    .filter(**filter_args)
                    .aggregate(
                        _order__max=Coalesce(
                            ExpressionWrapper(
                                Max("_order") + Value(1), output_field=IntegerField()
                            ),
                            Value(0),
                        ),
                    )["_order__max"]
                )
            fields = meta.local_concrete_fields
            if not pk_set:
                fields = [f for f in fields if f is not meta.auto_field]

            returning_fields = meta.db_returning_fields
            results = self._do_insert(
                cls._base_manager, using, fields, returning_fields, raw
            )
            if results:
                for value, field in zip(results[0], returning_fields):
                    setattr(self, field.attname, value)
        return updated
```
### 84 - django/db/migrations/operations/models.py:

Start line: 689, End line: 741

```python
class AlterModelOptions(ModelOptionOperation):
    """
    Set new model options that don't directly affect the database schema
    (like verbose_name, permissions, ordering). Python code in migrations
    may still need them.
    """

    # Model options we want to compare and preserve in an AlterModelOptions op
    ALTER_OPTION_KEYS = [
        "base_manager_name",
        "default_manager_name",
        "default_related_name",
        "get_latest_by",
        "managed",
        "ordering",
        "permissions",
        "default_permissions",
        "select_on_save",
        "verbose_name",
        "verbose_name_plural",
    ]

    def __init__(self, name, options):
        self.options = options
        super().__init__(name)

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            "options": self.options,
        }
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.alter_model_options(
            app_label,
            self.name_lower,
            self.options,
            self.ALTER_OPTION_KEYS,
        )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        pass

    def describe(self):
        return "Change Meta options on %s" % self.name

    @property
    def migration_name_fragment(self):
        return "alter_%s_options" % self.name_lower
```
### 87 - django/db/backends/mysql/schema.py:

Start line: 120, End line: 135

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def remove_constraint(self, model, constraint):
        if isinstance(constraint, UniqueConstraint):
            self._create_missing_fk_index(
                model,
                fields=constraint.fields,
                expressions=constraint.expressions,
            )
        super().remove_constraint(model, constraint)

    def remove_index(self, model, index):
        self._create_missing_fk_index(
            model,
            fields=[field_name for field_name, _ in index.fields_orders],
            expressions=index.expressions,
        )
        super().remove_index(model, index)
```
### 89 - django/db/backends/base/schema.py:

Start line: 564, End line: 596

```python
class BaseDatabaseSchemaEditor:

    def _delete_composed_index(self, model, fields, constraint_kwargs, sql):
        meta_constraint_names = {
            constraint.name for constraint in model._meta.constraints
        }
        meta_index_names = {constraint.name for constraint in model._meta.indexes}
        columns = [model._meta.get_field(field).column for field in fields]
        constraint_names = self._constraint_names(
            model,
            columns,
            exclude=meta_constraint_names | meta_index_names,
            **constraint_kwargs,
        )
        if (
            constraint_kwargs.get("unique") is True
            and constraint_names
            and self.connection.features.allows_multiple_constraints_on_same_fields
        ):
            # Constraint matching the unique_together name.
            default_name = str(
                self._unique_constraint_name(model._meta.db_table, columns, quote=False)
            )
            if default_name in constraint_names:
                constraint_names = [default_name]
        if len(constraint_names) != 1:
            raise ValueError(
                "Found wrong number (%s) of constraints for %s(%s)"
                % (
                    len(constraint_names),
                    model._meta.db_table,
                    ", ".join(columns),
                )
            )
        self.execute(self._delete_constraint_sql(sql, model, constraint_names[0]))
```
### 94 - django/db/backends/postgresql/introspection.py:

Start line: 1, End line: 38

```python
from collections import namedtuple

from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index

FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield",))


class DatabaseIntrospection(BaseDatabaseIntrospection):
    # Maps type codes to Django Field types.
    data_types_reverse = {
        16: "BooleanField",
        17: "BinaryField",
        20: "BigIntegerField",
        21: "SmallIntegerField",
        23: "IntegerField",
        25: "TextField",
        700: "FloatField",
        701: "FloatField",
        869: "GenericIPAddressField",
        1042: "CharField",  # blank-padded
        1043: "CharField",
        1082: "DateField",
        1083: "TimeField",
        1114: "DateTimeField",
        1184: "DateTimeField",
        1186: "DurationField",
        1266: "TimeField",
        1700: "DecimalField",
        2950: "UUIDField",
        3802: "JSONField",
    }
    # A hook for subclasses.
    index_default_access_method = "btree"

    ignored_tables = []
    # ... other code
```
### 95 - django/db/backends/base/schema.py:

Start line: 398, End line: 420

```python
class BaseDatabaseSchemaEditor:

    @staticmethod
    def _effective_default(field):
        # This method allows testing its logic without a connection.
        if field.has_default():
            default = field.get_default()
        elif not field.null and field.blank and field.empty_strings_allowed:
            if field.get_internal_type() == "BinaryField":
                default = b""
            else:
                default = ""
        elif getattr(field, "auto_now", False) or getattr(field, "auto_now_add", False):
            internal_type = field.get_internal_type()
            if internal_type == "DateTimeField":
                default = timezone.now()
            else:
                default = datetime.now()
                if internal_type == "DateField":
                    default = default.date()
                elif internal_type == "TimeField":
                    default = default.time()
        else:
            default = None
        return default
```
### 96 - django/core/management/commands/inspectdb.py:

Start line: 255, End line: 313

```python
class Command(BaseCommand):

    def normalize_col_name(self, col_name, used_column_names, is_relation):
        """
        Modify the column name to make it Python-compatible as a field name
        """
        field_params = {}
        field_notes = []

        new_name = col_name.lower()
        if new_name != col_name:
            field_notes.append("Field name made lowercase.")

        if is_relation:
            if new_name.endswith("_id"):
                new_name = new_name[:-3]
            else:
                field_params["db_column"] = col_name

        new_name, num_repl = re.subn(r"\W", "_", new_name)
        if num_repl > 0:
            field_notes.append("Field renamed to remove unsuitable characters.")

        if new_name.find(LOOKUP_SEP) >= 0:
            while new_name.find(LOOKUP_SEP) >= 0:
                new_name = new_name.replace(LOOKUP_SEP, "_")
            if col_name.lower().find(LOOKUP_SEP) >= 0:
                # Only add the comment if the double underscore was in the original name
                field_notes.append(
                    "Field renamed because it contained more than one '_' in a row."
                )

        if new_name.startswith("_"):
            new_name = "field%s" % new_name
            field_notes.append("Field renamed because it started with '_'.")

        if new_name.endswith("_"):
            new_name = "%sfield" % new_name
            field_notes.append("Field renamed because it ended with '_'.")

        if keyword.iskeyword(new_name):
            new_name += "_field"
            field_notes.append("Field renamed because it was a Python reserved word.")

        if new_name[0].isdigit():
            new_name = "number_%s" % new_name
            field_notes.append(
                "Field renamed because it wasn't a valid Python identifier."
            )

        if new_name in used_column_names:
            num = 0
            while "%s_%d" % (new_name, num) in used_column_names:
                num += 1
            new_name = "%s_%d" % (new_name, num)
            field_notes.append("Field renamed because of name conflict.")

        if col_name != new_name and field_notes:
            field_params["db_column"] = col_name

        return new_name, field_params, field_notes
```
### 97 - django/db/backends/oracle/features.py:

Start line: 1, End line: 86

```python
from django.db import DatabaseError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    minimum_database_version = (19,)
    # Oracle crashes with "ORA-00932: inconsistent datatypes: expected - got
    # BLOB" when grouping by LOBs (#24096).
    allows_group_by_lob = False
    allows_group_by_refs = False
    interprets_empty_strings_as_nulls = True
    has_select_for_update = True
    has_select_for_update_nowait = True
    has_select_for_update_skip_locked = True
    has_select_for_update_of = True
    select_for_update_of_column = True
    can_return_columns_from_insert = True
    supports_subqueries_in_group_by = False
    ignores_unnecessary_order_by_in_subqueries = False
    supports_transactions = True
    supports_timezones = False
    has_native_duration_field = True
    can_defer_constraint_checks = True
    supports_partially_nullable_unique_constraints = False
    supports_deferrable_unique_constraints = True
    truncates_names = True
    supports_tablespaces = True
    supports_sequence_reset = False
    can_introspect_materialized_views = True
    atomic_transactions = False
    nulls_order_largest = True
    requires_literal_defaults = True
    closed_cursor_error_class = InterfaceError
    bare_select_suffix = " FROM DUAL"
    # Select for update with limit can be achieved on Oracle, but not with the
    # current backend.
    supports_select_for_update_with_limit = False
    supports_temporal_subtraction = True
    # Oracle doesn't ignore quoted identifiers case but the current backend
    # does by uppercasing all identifiers.
    ignores_table_name_case = True
    supports_index_on_text_field = False
    create_test_procedure_without_params_sql = """
        CREATE PROCEDURE "TEST_PROCEDURE" AS
            V_I INTEGER;
        BEGIN
            V_I := 1;
        END;
    """
    create_test_procedure_with_int_param_sql = """
        CREATE PROCEDURE "TEST_PROCEDURE" (P_I INTEGER) AS
            V_I INTEGER;
        BEGIN
            V_I := P_I;
        END;
    """
    create_test_table_with_composite_primary_key = """
        CREATE TABLE test_table_composite_pk (
            column_1 NUMBER(11) NOT NULL,
            column_2 NUMBER(11) NOT NULL,
            PRIMARY KEY (column_1, column_2)
        )
    """
    supports_callproc_kwargs = True
    supports_over_clause = True
    supports_frame_range_fixed_distance = True
    supports_ignore_conflicts = False
    max_query_params = 2**16 - 1
    supports_partial_indexes = False
    can_rename_index = True
    supports_slicing_ordering_in_compound = True
    requires_compound_order_by_subquery = True
    allows_multiple_constraints_on_same_fields = False
    supports_boolean_expr_in_select_clause = False
    supports_comparing_boolean_expr = False
    supports_primitives_in_json_field = False
    supports_json_field_contains = False
    supports_collation_on_textfield = False
    test_collations = {
        "ci": "BINARY_CI",
        "cs": "BINARY",
        "non_default": "SWEDISH_CI",
        "swedish_ci": "SWEDISH_CI",
    }
    test_now_utc_template = "CURRENT_TIMESTAMP AT TIME ZONE 'UTC'"
```
### 100 - django/db/models/base.py:

Start line: 574, End line: 612

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def from_db(cls, db, field_names, values):
        if len(values) != len(cls._meta.concrete_fields):
            values_iter = iter(values)
            values = [
                next(values_iter) if f.attname in field_names else DEFERRED
                for f in cls._meta.concrete_fields
            ]
        new = cls(*values)
        new._state.adding = False
        new._state.db = db
        return new

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)

    def __str__(self):
        return "%s object (%s)" % (self.__class__.__name__, self.pk)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        if self._meta.concrete_model != other._meta.concrete_model:
            return False
        my_pk = self.pk
        if my_pk is None:
            return self is other
        return my_pk == other.pk

    def __hash__(self):
        if self.pk is None:
            raise TypeError("Model instances without primary key value are unhashable")
        return hash(self.pk)

    def __reduce__(self):
        data = self.__getstate__()
        data[DJANGO_VERSION_PICKLE_KEY] = django.__version__
        class_id = self._meta.app_label, self._meta.object_name
        return model_unpickle, (class_id,), data
```
### 101 - django/db/models/options.py:

Start line: 1, End line: 57

```python
import bisect
import copy
import inspect
import warnings
from collections import defaultdict

from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import connections
from django.db.models import AutoField, Manager, OrderWrt, UniqueConstraint
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override

PROXY_PARENTS = object()

EMPTY_RELATION_TREE = ()

IMMUTABLE_WARNING = (
    "The return type of '%s' should never be mutated. If you want to manipulate this "
    "list for your own use, make a copy first."
)

DEFAULT_NAMES = (
    "verbose_name",
    "verbose_name_plural",
    "db_table",
    "ordering",
    "unique_together",
    "permissions",
    "get_latest_by",
    "order_with_respect_to",
    "app_label",
    "db_tablespace",
    "abstract",
    "managed",
    "proxy",
    "swappable",
    "auto_created",
    # Must be kept for backward compatibility with old migrations.
    "index_together",
    "apps",
    "default_permissions",
    "select_on_save",
    "default_related_name",
    "required_db_features",
    "required_db_vendor",
    "base_manager_name",
    "default_manager_name",
    "indexes",
    "constraints",
)
```
### 104 - django/db/models/fields/__init__.py:

Start line: 2332, End line: 2366

```python
class TextField(Field):
    description = _("Text")

    def __init__(self, *args, db_collation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_collation = db_collation

    def check(self, **kwargs):
        databases = kwargs.get("databases") or []
        return [
            *super().check(**kwargs),
            *self._check_db_collation(databases),
        ]

    def _check_db_collation(self, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, self.model):
                continue
            connection = connections[db]
            if not (
                self.db_collation is None
                or "supports_collation_on_textfield"
                in self.model._meta.required_db_features
                or connection.features.supports_collation_on_textfield
            ):
                errors.append(
                    checks.Error(
                        "%s does not support a database collation on "
                        "TextFields." % connection.display_name,
                        obj=self,
                        id="fields.E190",
                    ),
                )
        return errors
```
### 105 - django/db/models/base.py:

Start line: 1350, End line: 1379

```python
class Model(AltersData, metaclass=ModelBase):

    def _perform_date_checks(self, date_checks):
        errors = {}
        for model_class, lookup_type, field, unique_for in date_checks:
            lookup_kwargs = {}
            # there's a ticket to add a date lookup, we can remove this special
            # case if that makes it's way in
            date = getattr(self, unique_for)
            if date is None:
                continue
            if lookup_type == "date":
                lookup_kwargs["%s__day" % unique_for] = date.day
                lookup_kwargs["%s__month" % unique_for] = date.month
                lookup_kwargs["%s__year" % unique_for] = date.year
            else:
                lookup_kwargs["%s__%s" % (unique_for, lookup_type)] = getattr(
                    date, lookup_type
                )
            lookup_kwargs[field] = getattr(self, field)

            qs = model_class._default_manager.filter(**lookup_kwargs)
            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            if not self._state.adding and self.pk is not None:
                qs = qs.exclude(pk=self.pk)

            if qs.exists():
                errors.setdefault(field, []).append(
                    self.date_error_message(lookup_type, field, unique_for)
                )
        return errors
```
### 107 - django/db/models/base.py:

Start line: 1123, End line: 1150

```python
class Model(AltersData, metaclass=ModelBase):

    def delete(self, using=None, keep_parents=False):
        if self.pk is None:
            raise ValueError(
                "%s object can't be deleted because its %s attribute is set "
                "to None." % (self._meta.object_name, self._meta.pk.attname)
            )
        using = using or router.db_for_write(self.__class__, instance=self)
        collector = Collector(using=using, origin=self)
        collector.collect([self], keep_parents=keep_parents)
        return collector.delete()

    delete.alters_data = True

    async def adelete(self, using=None, keep_parents=False):
        return await sync_to_async(self.delete)(
            using=using,
            keep_parents=keep_parents,
        )

    adelete.alters_data = True

    def _get_FIELD_display(self, field):
        value = getattr(self, field.attname)
        choices_dict = dict(make_hashable(field.flatchoices))
        # force_str() to coerce lazy strings.
        return force_str(
            choices_dict.get(make_hashable(value), value), strings_only=True
        )
```
### 108 - django/db/migrations/operations/__init__.py:

Start line: 1, End line: 43

```python
from .fields import AddField, AlterField, RemoveField, RenameField
from .models import (
    AddConstraint,
    AddIndex,
    AlterIndexTogether,
    AlterModelManagers,
    AlterModelOptions,
    AlterModelTable,
    AlterOrderWithRespectTo,
    AlterUniqueTogether,
    CreateModel,
    DeleteModel,
    RemoveConstraint,
    RemoveIndex,
    RenameIndex,
    RenameModel,
)
from .special import RunPython, RunSQL, SeparateDatabaseAndState

__all__ = [
    "CreateModel",
    "DeleteModel",
    "AlterModelTable",
    "AlterUniqueTogether",
    "RenameModel",
    "AlterIndexTogether",
    "AlterModelOptions",
    "AddIndex",
    "RemoveIndex",
    "RenameIndex",
    "AddField",
    "RemoveField",
    "AlterField",
    "RenameField",
    "AddConstraint",
    "RemoveConstraint",
    "SeparateDatabaseAndState",
    "RunSQL",
    "RunPython",
    "AlterOrderWithRespectTo",
    "AlterModelManagers",
]
```
### 109 - django/db/models/base.py:

Start line: 459, End line: 572

```python
class Model(AltersData, metaclass=ModelBase):
    def __init__(self, *args, **kwargs):
        # Alias some things as locals to avoid repeat global lookups
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED
        if opts.abstract:
            raise TypeError("Abstract models cannot be instantiated.")

        pre_init.send(sender=cls, args=args, kwargs=kwargs)

        # Set up the storage for instance state
        self._state = ModelState()

        # There is a rather weird disparity here; if kwargs, it's set, then args
        # overrides it. It should be one or the other; don't duplicate the work
        # The reason for the kwargs check is that standard iterator passes in by
        # args, and instantiation for iteration is 33% faster.
        if len(args) > len(opts.concrete_fields):
            # Daft, but matches old exception sans the err msg.
            raise IndexError("Number of args exceeds number of fields")

        if not kwargs:
            fields_iter = iter(opts.concrete_fields)
            # The ordering of the zip calls matter - zip throws StopIteration
            # when an iter throws it. So if the first iter throws it, the second
            # is *not* consumed. We rely on this, so don't change the order
            # without changing the logic.
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
        else:
            # Slower, kwargs-ready version.
            fields_iter = iter(opts.fields)
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
                if kwargs.pop(field.name, NOT_PROVIDED) is not NOT_PROVIDED:
                    raise TypeError(
                        f"{cls.__qualname__}() got both positional and "
                        f"keyword arguments for field '{field.name}'."
                    )

        # Now we're left with the unprocessed fields that *must* come from
        # keywords, or default.

        for field in fields_iter:
            is_related_object = False
            # Virtual field
            if field.attname not in kwargs and field.column is None:
                continue
            if kwargs:
                if isinstance(field.remote_field, ForeignObjectRel):
                    try:
                        # Assume object instance was passed in.
                        rel_obj = kwargs.pop(field.name)
                        is_related_object = True
                    except KeyError:
                        try:
                            # Object instance wasn't passed in -- must be an ID.
                            val = kwargs.pop(field.attname)
                        except KeyError:
                            val = field.get_default()
                else:
                    try:
                        val = kwargs.pop(field.attname)
                    except KeyError:
                        # This is done with an exception rather than the
                        # default argument on pop because we don't want
                        # get_default() to be evaluated, and then not used.
                        # Refs #12057.
                        val = field.get_default()
            else:
                val = field.get_default()

            if is_related_object:
                # If we are passed a related instance, set it using the
                # field.name instead of field.attname (e.g. "user" instead of
                # "user_id") so that the object gets properly cached (and type
                # checked) by the RelatedObjectDescriptor.
                if rel_obj is not _DEFERRED:
                    _setattr(self, field.name, rel_obj)
            else:
                if val is not _DEFERRED:
                    _setattr(self, field.attname, val)

        if kwargs:
            property_names = opts._property_names
            unexpected = ()
            for prop, value in kwargs.items():
                # Any remaining kwargs must correspond to properties or virtual
                # fields.
                if prop in property_names:
                    if value is not _DEFERRED:
                        _setattr(self, prop, value)
                else:
                    try:
                        opts.get_field(prop)
                    except FieldDoesNotExist:
                        unexpected += (prop,)
                    else:
                        if value is not _DEFERRED:
                            _setattr(self, prop, value)
            if unexpected:
                unexpected_names = ", ".join(repr(n) for n in unexpected)
                raise TypeError(
                    f"{cls.__name__}() got unexpected keyword arguments: "
                    f"{unexpected_names}"
                )
        super().__init__()
        post_init.send(sender=cls, instance=self)
```
### 114 - django/core/management/commands/inspectdb.py:

Start line: 355, End line: 395

```python
class Command(BaseCommand):

    def get_meta(
        self, table_name, constraints, column_to_field_name, is_view, is_partition
    ):
        """
        Return a sequence comprising the lines of code necessary
        to construct the inner Meta class for the model corresponding
        to the given database table name.
        """
        unique_together = []
        has_unsupported_constraint = False
        for params in constraints.values():
            if params["unique"]:
                columns = params["columns"]
                if None in columns:
                    has_unsupported_constraint = True
                columns = [
                    x for x in columns if x is not None and x in column_to_field_name
                ]
                if len(columns) > 1:
                    unique_together.append(
                        str(tuple(column_to_field_name[c] for c in columns))
                    )
        if is_view:
            managed_comment = "  # Created from a view. Don't remove."
        elif is_partition:
            managed_comment = "  # Created from a partition. Don't remove."
        else:
            managed_comment = ""
        meta = [""]
        if has_unsupported_constraint:
            meta.append("    # A unique constraint could not be introspected.")
        meta += [
            "    class Meta:",
            "        managed = False%s" % managed_comment,
            "        db_table = %r" % table_name,
        ]
        if unique_together:
            tup = "(" + ", ".join(unique_together) + ",)"
            meta += ["        unique_together = %s" % tup]
        return meta
```
### 119 - django/db/backends/mysql/features.py:

Start line: 63, End line: 81

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def test_collations(self):
        charset = "utf8"
        if (
            self.connection.mysql_is_mariadb
            and self.connection.mysql_version >= (10, 6)
        ) or (
            not self.connection.mysql_is_mariadb
            and self.connection.mysql_version >= (8, 0, 30)
        ):
            # utf8 is an alias for utf8mb3 in MariaDB 10.6+ and MySQL 8.0.30+.
            charset = "utf8mb3"
        return {
            "ci": f"{charset}_general_ci",
            "non_default": f"{charset}_esperanto_ci",
            "swedish_ci": f"{charset}_swedish_ci",
        }

    test_now_utc_template = "UTC_TIMESTAMP(6)"
```
### 120 - django/db/backends/base/schema.py:

Start line: 368, End line: 396

```python
class BaseDatabaseSchemaEditor:

    def skip_default(self, field):
        """
        Some backends don't accept default values for certain columns types
        (i.e. MySQL longtext and longblob).
        """
        return False

    def skip_default_on_alter(self, field):
        """
        Some backends don't accept default values for certain columns types
        (i.e. MySQL longtext and longblob) in the ALTER COLUMN statement.
        """
        return False

    def prepare_default(self, value):
        """
        Only used for backends which have requires_literal_defaults feature
        """
        raise NotImplementedError(
            "subclasses of BaseDatabaseSchemaEditor for backends which have "
            "requires_literal_defaults must provide a prepare_default() method"
        )

    def _column_default_sql(self, field):
        """
        Return the SQL to use in a DEFAULT clause. The resulting string should
        contain a '%s' placeholder for a default value.
        """
        return "%s"
```
### 121 - django/db/backends/mysql/introspection.py:

Start line: 1, End line: 19

```python
from collections import namedtuple

import sqlparse
from MySQLdb.constants import FIELD_TYPE

from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index
from django.utils.datastructures import OrderedSet

FieldInfo = namedtuple(
    "FieldInfo", BaseFieldInfo._fields + ("extra", "is_unsigned", "has_json_constraint")
)
InfoLine = namedtuple(
    "InfoLine",
    "col_name data_type max_len num_prec num_scale extra column_default "
    "collation is_unsigned",
)
```
### 123 - django/db/backends/base/features.py:

Start line: 223, End line: 366

```python
class BaseDatabaseFeatures:
    minimum_database_version =

    # Can the backend clone databases for parallel test execution?
    # Defaults to False to allow third-party backends to opt-in.
    can_clone_databases = False

    # Does the backend consider table names with different casing to
    # be equal?
    ignores_table_name_case = False

    # Place FOR UPDATE right after FROM clause. Used on MSSQL.
    for_update_after_from = False

    # Combinatorial flags
    supports_select_union = True
    supports_select_intersection = True
    supports_select_difference = True
    supports_slicing_ordering_in_compound = False
    supports_parentheses_in_compound = True
    requires_compound_order_by_subquery = False

    # Does the database support SQL 2003 FILTER (WHERE ...) in aggregate
    # expressions?
    supports_aggregate_filter_clause = False

    # Does the backend support indexing a TextField?
    supports_index_on_text_field = True

    # Does the backend support window expressions (expression OVER (...))?
    supports_over_clause = False
    supports_frame_range_fixed_distance = False
    only_supports_unbounded_with_preceding_and_following = False

    # Does the backend support CAST with precision?
    supports_cast_with_precision = True

    # How many second decimals does the database return when casting a value to
    # a type with time?
    time_cast_precision = 6

    # SQL to create a procedure for use by the Django test suite. The
    # functionality of the procedure isn't important.
    create_test_procedure_without_params_sql = None
    create_test_procedure_with_int_param_sql = None

    # SQL to create a table with a composite primary key for use by the Django
    # test suite.
    create_test_table_with_composite_primary_key = None

    # Does the backend support keyword parameters for cursor.callproc()?
    supports_callproc_kwargs = False

    # What formats does the backend EXPLAIN syntax support?
    supported_explain_formats = set()

    # Does the backend support the default parameter in lead() and lag()?
    supports_default_in_lead_lag = True

    # Does the backend support ignoring constraint or uniqueness errors during
    # INSERT?
    supports_ignore_conflicts = True
    # Does the backend support updating rows on constraint or uniqueness errors
    # during INSERT?
    supports_update_conflicts = False
    supports_update_conflicts_with_target = False

    # Does this backend require casting the results of CASE expressions used
    # in UPDATE statements to ensure the expression has the correct type?
    requires_casted_case_in_updates = False

    # Does the backend support partial indexes (CREATE INDEX ... WHERE ...)?
    supports_partial_indexes = True
    supports_functions_in_partial_indexes = True
    # Does the backend support covering indexes (CREATE INDEX ... INCLUDE ...)?
    supports_covering_indexes = False
    # Does the backend support indexes on expressions?
    supports_expression_indexes = True
    # Does the backend treat COLLATE as an indexed expression?
    collate_as_index_expression = False

    # Does the database allow more than one constraint or index on the same
    # field(s)?
    allows_multiple_constraints_on_same_fields = True

    # Does the backend support boolean expressions in SELECT and GROUP BY
    # clauses?
    supports_boolean_expr_in_select_clause = True
    # Does the backend support comparing boolean expressions in WHERE clauses?
    # Eg: WHERE (price > 0) IS NOT NULL
    supports_comparing_boolean_expr = True

    # Does the backend support JSONField?
    supports_json_field = True
    # Can the backend introspect a JSONField?
    can_introspect_json_field = True
    # Does the backend support primitives in JSONField?
    supports_primitives_in_json_field = True
    # Is there a true datatype for JSON?
    has_native_json_field = False
    # Does the backend use PostgreSQL-style JSON operators like '->'?
    has_json_operators = False
    # Does the backend support __contains and __contained_by lookups for
    # a JSONField?
    supports_json_field_contains = True
    # Does value__d__contains={'f': 'g'} (without a list around the dict) match
    # {'d': [{'f': 'g'}]}?
    json_key_contains_list_matching_requires_list = False
    # Does the backend support JSONObject() database function?
    has_json_object_function = True

    # Does the backend support column collations?
    supports_collation_on_charfield = True
    supports_collation_on_textfield = True
    # Does the backend support non-deterministic collations?
    supports_non_deterministic_collations = True

    # Does the backend support the logical XOR operator?
    supports_logical_xor = False

    # Set to (exception, message) if null characters in text are disallowed.
    prohibits_null_characters_in_text_exception = None

    # Collation names for use by the Django test suite.
    test_collations = {
        "ci": None,  # Case-insensitive.
        "cs": None,  # Case-sensitive.
        "non_default": None,  # Non-default.
        "swedish_ci": None,  # Swedish case-insensitive.
    }
    # SQL template override for tests.aggregation.tests.NowUTC
    test_now_utc_template = None

    # A set of dotted paths to tests in Django's test suite that are expected
    # to fail on this database.
    django_test_expected_failures = set()
    # A map of reasons to sets of dotted paths to tests in Django's test suite
    # that should be skipped for this database.
    django_test_skips = {}

    def __init__(self, connection):
        self.connection = connection

    @cached_property
    def supports_explaining_query_execution(self):
        """Does this backend support explaining query execution?"""
        return self.connection.ops.explain_prefix is not None
    # ... other code
```
### 125 - django/db/backends/mysql/introspection.py:

Start line: 22, End line: 45

```python
class DatabaseIntrospection(BaseDatabaseIntrospection):
    data_types_reverse = {
        FIELD_TYPE.BLOB: "TextField",
        FIELD_TYPE.CHAR: "CharField",
        FIELD_TYPE.DECIMAL: "DecimalField",
        FIELD_TYPE.NEWDECIMAL: "DecimalField",
        FIELD_TYPE.DATE: "DateField",
        FIELD_TYPE.DATETIME: "DateTimeField",
        FIELD_TYPE.DOUBLE: "FloatField",
        FIELD_TYPE.FLOAT: "FloatField",
        FIELD_TYPE.INT24: "IntegerField",
        FIELD_TYPE.JSON: "JSONField",
        FIELD_TYPE.LONG: "IntegerField",
        FIELD_TYPE.LONGLONG: "BigIntegerField",
        FIELD_TYPE.SHORT: "SmallIntegerField",
        FIELD_TYPE.STRING: "CharField",
        FIELD_TYPE.TIME: "TimeField",
        FIELD_TYPE.TIMESTAMP: "DateTimeField",
        FIELD_TYPE.TINY: "IntegerField",
        FIELD_TYPE.TINY_BLOB: "TextField",
        FIELD_TYPE.MEDIUM_BLOB: "TextField",
        FIELD_TYPE.LONG_BLOB: "TextField",
        FIELD_TYPE.VAR_STRING: "CharField",
    }
```
### 126 - django/db/migrations/operations/models.py:

Start line: 41, End line: 111

```python
class CreateModel(ModelOperation):
    """Create a model's table."""

    serialization_expand_args = ["fields", "options", "managers"]

    def __init__(self, name, fields, options=None, bases=None, managers=None):
        self.fields = fields
        self.options = options or {}
        self.bases = bases or (models.Model,)
        self.managers = managers or []
        super().__init__(name)
        # Sanity-check that there are no duplicated field names, bases, or
        # manager names
        _check_for_duplicates("fields", (name for name, _ in self.fields))
        _check_for_duplicates(
            "bases",
            (
                base._meta.label_lower
                if hasattr(base, "_meta")
                else base.lower()
                if isinstance(base, str)
                else base
                for base in self.bases
            ),
        )
        _check_for_duplicates("managers", (name for name, _ in self.managers))

    def deconstruct(self):
        kwargs = {
            "name": self.name,
            "fields": self.fields,
        }
        if self.options:
            kwargs["options"] = self.options
        if self.bases and self.bases != (models.Model,):
            kwargs["bases"] = self.bases
        if self.managers and self.managers != [("objects", models.Manager())]:
            kwargs["managers"] = self.managers
        return (self.__class__.__qualname__, [], kwargs)

    def state_forwards(self, app_label, state):
        state.add_model(
            ModelState(
                app_label,
                self.name,
                list(self.fields),
                dict(self.options),
                tuple(self.bases),
                list(self.managers),
            )
        )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.create_model(model)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.delete_model(model)

    def describe(self):
        return "Create %smodel %s" % (
            "proxy " if self.options.get("proxy", False) else "",
            self.name,
        )

    @property
    def migration_name_fragment(self):
        return self.name_lower
```
### 127 - django/db/backends/base/schema.py:

Start line: 1135, End line: 1162

```python
class BaseDatabaseSchemaEditor:

    def _alter_column_null_sql(self, model, old_field, new_field):
        """
        Hook to specialize column null alteration.

        Return a (sql, params) fragment to set a column to null or non-null
        as required by new_field, or None if no changes are required.
        """
        if (
            self.connection.features.interprets_empty_strings_as_nulls
            and new_field.empty_strings_allowed
        ):
            # The field is nullable in the database anyway, leave it alone.
            return
        else:
            new_db_params = new_field.db_parameters(connection=self.connection)
            sql = (
                self.sql_alter_column_null
                if new_field.null
                else self.sql_alter_column_not_null
            )
            return (
                sql
                % {
                    "column": self.quote_name(new_field.column),
                    "type": new_db_params["type"],
                },
                [],
            )
```
### 129 - django/db/backends/postgresql/introspection.py:

Start line: 265, End line: 297

```python
class DatabaseIntrospection(BaseDatabaseIntrospection):
    data_types_reverse =

    def get_constraints(self, cursor, table_name):
        # ... other code
        for (
            index,
            columns,
            unique,
            primary,
            orders,
            type_,
            definition,
            options,
        ) in cursor.fetchall():
            if index not in constraints:
                basic_index = (
                    type_ == self.index_default_access_method
                    and
                    # '_btree' references
                    # django.contrib.postgres.indexes.BTreeIndex.suffix.
                    not index.endswith("_btree")
                    and options is None
                )
                constraints[index] = {
                    "columns": columns if columns != [None] else [],
                    "orders": orders if orders != [None] else [],
                    "primary_key": primary,
                    "unique": unique,
                    "foreign_key": None,
                    "check": False,
                    "index": True,
                    "type": Index.suffix if basic_index else type_,
                    "definition": definition,
                    "options": options,
                }
        return constraints
```
### 131 - django/db/models/options.py:

Start line: 441, End line: 463

```python
class Options:

    @cached_property
    def managers(self):
        managers = []
        seen_managers = set()
        bases = (b for b in self.model.mro() if hasattr(b, "_meta"))
        for depth, base in enumerate(bases):
            for manager in base._meta.local_managers:
                if manager.name in seen_managers:
                    continue

                manager = copy.copy(manager)
                manager.model = self.model
                seen_managers.add(manager.name)
                managers.append((depth, manager.creation_counter, manager))

        return make_immutable_fields_list(
            "managers",
            (m[2] for m in sorted(managers)),
        )

    @cached_property
    def managers_map(self):
        return {manager.name: manager for manager in self.managers}
```
### 132 - django/db/migrations/autodetector.py:

Start line: 1534, End line: 1553

```python
class MigrationAutodetector:

    def generate_altered_db_table(self):
        models_to_check = self.kept_model_keys.union(
            self.kept_proxy_keys, self.kept_unmanaged_keys
        )
        for app_label, model_name in sorted(models_to_check):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            old_db_table_name = old_model_state.options.get("db_table")
            new_db_table_name = new_model_state.options.get("db_table")
            if old_db_table_name != new_db_table_name:
                self.add_operation(
                    app_label,
                    operations.AlterModelTable(
                        name=model_name,
                        table=new_db_table_name,
                    ),
                )
```
### 134 - django/db/models/base.py:

Start line: 653, End line: 673

```python
class Model(AltersData, metaclass=ModelBase):

    def _get_pk_val(self, meta=None):
        meta = meta or self._meta
        return getattr(self, meta.pk.attname)

    def _set_pk_val(self, value):
        for parent_link in self._meta.parents.values():
            if parent_link and parent_link != self._meta.pk:
                setattr(self, parent_link.target_field.attname, value)
        return setattr(self, self._meta.pk.attname, value)

    pk = property(_get_pk_val, _set_pk_val)

    def get_deferred_fields(self):
        """
        Return a set containing names of deferred fields for this instance.
        """
        return {
            f.attname
            for f in self._meta.concrete_fields
            if f.attname not in self.__dict__
        }
```
### 136 - django/db/models/options.py:

Start line: 287, End line: 329

```python
class Options:

    def _prepare(self, model):
        if self.order_with_respect_to:
            # The app registry will not be ready at this point, so we cannot
            # use get_field().
            query = self.order_with_respect_to
            try:
                self.order_with_respect_to = next(
                    f
                    for f in self._get_fields(reverse=False)
                    if f.name == query or f.attname == query
                )
            except StopIteration:
                raise FieldDoesNotExist(
                    "%s has no field named '%s'" % (self.object_name, query)
                )

            self.ordering = ("_order",)
            if not any(
                isinstance(field, OrderWrt) for field in model._meta.local_fields
            ):
                model.add_to_class("_order", OrderWrt())
        else:
            self.order_with_respect_to = None

        if self.pk is None:
            if self.parents:
                # Promote the first parent link in lieu of adding yet another
                # field.
                field = next(iter(self.parents.values()))
                # Look for a local field with the same name as the
                # first parent link. If a local field has already been
                # created, use it instead of promoting the parent
                already_created = [
                    fld for fld in self.local_fields if fld.name == field.name
                ]
                if already_created:
                    field = already_created[0]
                field.primary_key = True
                self.setup_pk(field)
            else:
                pk_class = self._get_default_pk_class()
                auto = pk_class(verbose_name="ID", primary_key=True, auto_created=True)
                model.add_to_class("id", auto)
```
### 137 - django/db/backends/oracle/introspection.py:

Start line: 1, End line: 49

```python
from collections import namedtuple

import cx_Oracle

from django.db import models
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.utils.functional import cached_property

FieldInfo = namedtuple("FieldInfo", BaseFieldInfo._fields + ("is_autofield", "is_json"))


class DatabaseIntrospection(BaseDatabaseIntrospection):
    cache_bust_counter = 1

    # Maps type objects to Django Field types.
    @cached_property
    def data_types_reverse(self):
        if self.connection.cx_oracle_version < (8,):
            return {
                cx_Oracle.BLOB: "BinaryField",
                cx_Oracle.CLOB: "TextField",
                cx_Oracle.DATETIME: "DateField",
                cx_Oracle.FIXED_CHAR: "CharField",
                cx_Oracle.FIXED_NCHAR: "CharField",
                cx_Oracle.INTERVAL: "DurationField",
                cx_Oracle.NATIVE_FLOAT: "FloatField",
                cx_Oracle.NCHAR: "CharField",
                cx_Oracle.NCLOB: "TextField",
                cx_Oracle.NUMBER: "DecimalField",
                cx_Oracle.STRING: "CharField",
                cx_Oracle.TIMESTAMP: "DateTimeField",
            }
        else:
            return {
                cx_Oracle.DB_TYPE_DATE: "DateField",
                cx_Oracle.DB_TYPE_BINARY_DOUBLE: "FloatField",
                cx_Oracle.DB_TYPE_BLOB: "BinaryField",
                cx_Oracle.DB_TYPE_CHAR: "CharField",
                cx_Oracle.DB_TYPE_CLOB: "TextField",
                cx_Oracle.DB_TYPE_INTERVAL_DS: "DurationField",
                cx_Oracle.DB_TYPE_NCHAR: "CharField",
                cx_Oracle.DB_TYPE_NCLOB: "TextField",
                cx_Oracle.DB_TYPE_NVARCHAR: "CharField",
                cx_Oracle.DB_TYPE_NUMBER: "DecimalField",
                cx_Oracle.DB_TYPE_TIMESTAMP: "DateTimeField",
                cx_Oracle.DB_TYPE_VARCHAR: "CharField",
            }
```
### 138 - django/db/models/base.py:

Start line: 1998, End line: 2051

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_local_fields(cls, fields, option):
        from django.db import models

        # In order to avoid hitting the relation tree prematurely, we use our
        # own fields_map instead of using get_field()
        forward_fields_map = {}
        for field in cls._meta._get_fields(reverse=False):
            forward_fields_map[field.name] = field
            if hasattr(field, "attname"):
                forward_fields_map[field.attname] = field

        errors = []
        for field_name in fields:
            try:
                field = forward_fields_map[field_name]
            except KeyError:
                errors.append(
                    checks.Error(
                        "'%s' refers to the nonexistent field '%s'."
                        % (
                            option,
                            field_name,
                        ),
                        obj=cls,
                        id="models.E012",
                    )
                )
            else:
                if isinstance(field.remote_field, models.ManyToManyRel):
                    errors.append(
                        checks.Error(
                            "'%s' refers to a ManyToManyField '%s', but "
                            "ManyToManyFields are not permitted in '%s'."
                            % (
                                option,
                                field_name,
                                option,
                            ),
                            obj=cls,
                            id="models.E013",
                        )
                    )
                elif field not in cls._meta.local_fields:
                    errors.append(
                        checks.Error(
                            "'%s' refers to field '%s' which is not local to model "
                            "'%s'." % (option, field_name, cls._meta.object_name),
                            hint="This issue may be caused by multi-table inheritance.",
                            obj=cls,
                            id="models.E016",
                        )
                    )
        return errors
```
### 140 - django/db/backends/mysql/introspection.py:

Start line: 313, End line: 337

```python
class DatabaseIntrospection(BaseDatabaseIntrospection):

    def get_constraints(self, cursor, table_name):
        # ... other code
        for table, non_unique, index, colseq, column, order, type_ in [
            x[:6] + (x[10],) for x in cursor.fetchall()
        ]:
            if index not in constraints:
                constraints[index] = {
                    "columns": OrderedSet(),
                    "primary_key": False,
                    "unique": not non_unique,
                    "check": False,
                    "foreign_key": None,
                }
                if self.connection.features.supports_index_column_ordering:
                    constraints[index]["orders"] = []
            constraints[index]["index"] = True
            constraints[index]["type"] = (
                Index.suffix if type_ == "BTREE" else type_.lower()
            )
            constraints[index]["columns"].add(column)
            if self.connection.features.supports_index_column_ordering:
                constraints[index]["orders"].append("DESC" if order == "D" else "ASC")
        # Convert the sorted sets to lists
        for constraint in constraints.values():
            constraint["columns"] = list(constraint["columns"])
        return constraints
```
### 142 - django/db/models/base.py:

Start line: 1196, End line: 1236

```python
class Model(AltersData, metaclass=ModelBase):

    def _get_field_value_map(self, meta, exclude=None):
        if exclude is None:
            exclude = set()
        meta = meta or self._meta
        return {
            field.name: Value(getattr(self, field.attname), field)
            for field in meta.local_concrete_fields
            if field.name not in exclude
        }

    def prepare_database_save(self, field):
        if self.pk is None:
            raise ValueError(
                "Unsaved model instance %r cannot be used in an ORM query." % self
            )
        return getattr(self, field.remote_field.get_related_field().attname)

    def clean(self):
        """
        Hook for doing any extra model-wide validation after clean() has been
        called on every field by self.clean_fields. Any ValidationError raised
        by this method will not be associated with a particular field; it will
        have a special-case association with the field defined by NON_FIELD_ERRORS.
        """
        pass

    def validate_unique(self, exclude=None):
        """
        Check unique constraints on the model and raise ValidationError if any
        failed.
        """
        unique_checks, date_checks = self._get_unique_checks(exclude=exclude)

        errors = self._perform_unique_checks(unique_checks)
        date_errors = self._perform_date_checks(date_checks)

        for k, v in date_errors.items():
            errors.setdefault(k, []).extend(v)

        if errors:
            raise ValidationError(errors)
```
### 143 - django/db/models/base.py:

Start line: 1563, End line: 1593

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_default_pk(cls):
        if (
            not cls._meta.abstract
            and cls._meta.pk.auto_created
            and
            # Inherited PKs are checked in parents models.
            not (
                isinstance(cls._meta.pk, OneToOneField)
                and cls._meta.pk.remote_field.parent_link
            )
            and not settings.is_overridden("DEFAULT_AUTO_FIELD")
            and cls._meta.app_config
            and not cls._meta.app_config._is_default_auto_field_overridden
        ):
            return [
                checks.Warning(
                    f"Auto-created primary key used when not defining a "
                    f"primary key type, by default "
                    f"'{settings.DEFAULT_AUTO_FIELD}'.",
                    hint=(
                        f"Configure the DEFAULT_AUTO_FIELD setting or the "
                        f"{cls._meta.app_config.__class__.__qualname__}."
                        f"default_auto_field attribute to point to a subclass "
                        f"of AutoField, e.g. 'django.db.models.BigAutoField'."
                    ),
                    obj=cls,
                    id="models.W042",
                ),
            ]
        return []
```
### 144 - django/db/backends/base/features.py:

Start line: 1, End line: 385

```python
from django.db import ProgrammingError
from django.utils.functional import cached_property


class BaseDatabaseFeatures:
```
### 145 - django/db/models/fields/related.py:

Start line: 901, End line: 990

```python
class ForeignKey(ForeignObject):
    """
    Provide a many-to-one relation by adding a column to the local model
    to hold the remote value.

    By default ForeignKey will target the pk of the remote model but this
    behavior can be changed by using the ``to_field`` argument.
    """

    descriptor_class = ForeignKeyDeferredAttribute
    # Field flags
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False

    rel_class = ManyToOneRel

    empty_strings_allowed = False
    default_error_messages = {
        "invalid": _("%(model)s instance with %(field)s %(value)r does not exist.")
    }
    description = _("Foreign Key (type determined by related field)")

    def __init__(
        self,
        to,
        on_delete,
        related_name=None,
        related_query_name=None,
        limit_choices_to=None,
        parent_link=False,
        to_field=None,
        db_constraint=True,
        **kwargs,
    ):
        try:
            to._meta.model_name
        except AttributeError:
            if not isinstance(to, str):
                raise TypeError(
                    "%s(%r) is invalid. First parameter to ForeignKey must be "
                    "either a model, a model name, or the string %r"
                    % (
                        self.__class__.__name__,
                        to,
                        RECURSIVE_RELATIONSHIP_CONSTANT,
                    )
                )
        else:
            # For backwards compatibility purposes, we need to *try* and set
            # the to_field during FK construction. It won't be guaranteed to
            # be correct until contribute_to_class is called. Refs #12190.
            to_field = to_field or (to._meta.pk and to._meta.pk.name)
        if not callable(on_delete):
            raise TypeError("on_delete must be callable.")

        kwargs["rel"] = self.rel_class(
            self,
            to,
            to_field,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            parent_link=parent_link,
            on_delete=on_delete,
        )
        kwargs.setdefault("db_index", True)

        super().__init__(
            to,
            on_delete,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            from_fields=[RECURSIVE_RELATIONSHIP_CONSTANT],
            to_fields=[to_field],
            **kwargs,
        )
        self.db_constraint = db_constraint

    def __class_getitem__(cls, *args, **kwargs):
        return cls

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_on_delete(),
            *self._check_unique(),
        ]
```
### 146 - django/db/models/options.py:

Start line: 331, End line: 367

```python
class Options:

    def add_manager(self, manager):
        self.local_managers.append(manager)
        self._expire_cache()

    def add_field(self, field, private=False):
        # Insert the given field in the order in which it was created, using
        # the "creation_counter" attribute of the field.
        # Move many-to-many related fields from self.fields into
        # self.many_to_many.
        if private:
            self.private_fields.append(field)
        elif field.is_relation and field.many_to_many:
            bisect.insort(self.local_many_to_many, field)
        else:
            bisect.insort(self.local_fields, field)
            self.setup_pk(field)

        # If the field being added is a relation to another known field,
        # expire the cache on this field and the forward cache on the field
        # being referenced, because there will be new relationships in the
        # cache. Otherwise, expire the cache of references *to* this field.
        # The mechanism for getting at the related model is slightly odd -
        # ideally, we'd just ask for field.related_model. However, related_model
        # is a cached property, and all the models haven't been loaded yet, so
        # we need to make sure we don't cache a string reference.
        if (
            field.is_relation
            and hasattr(field.remote_field, "model")
            and field.remote_field.model
        ):
            try:
                field.remote_field.model._meta._expire_cache(forward=False)
            except AttributeError:
                pass
            self._expire_cache()
        else:
            self._expire_cache(reverse=False)
```
