(module . (_) @child.first @definition.module) @root

(decorated_definition [
    (function_definition) @check_child
    (class_definition) @check_child
  ]
) @root

(class_definition
  (identifier) @identifier
  (comment) @child.first @definition.class
  .
  (block (_) @child.last .)
) @root

(class_definition
  (identifier) @identifier
  (block . (_) @child.first)
) @root @definition.class

(function_definition
  (identifier) @identifier
  (comment) @child.first
  (block (_) @child.last .)
) @root @definition.function

(function_definition
  (identifier) @identifier
  (block
     . (_) @child.first
     (_) @child.last .)
) @root @definition.function

(function_definition
  (identifier) @identifier
  (block . (_) @child.first)
) @root @definition.function

(comment) @root @definition.comment @c

(import_statement) @root @definition.import
(future_import_statement) @root @definition.import
(import_from_statement) @root @definition.import

(block
  (_) @check_child
) @root

(_
  (block . (_) @child.first)
) @root @definition.statement
