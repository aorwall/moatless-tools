(
  (decorated_definition
    [
      (function_definition
        (identifier) @identifier
        (block) @no_children
      ) @definition.function
      (class_definition
        (identifier) @identifier
        (block) @no_children
      ) @definition.class
    ] .
  ) @root
  .
  (comment) @child.first @child.last
)

(
  (function_definition
    (identifier) @identifier
    (block) @no_children
  ) @root @definition.function
  .
  (comment) @child.first @child.last
)
