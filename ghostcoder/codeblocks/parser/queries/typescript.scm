(class_declaration
  [
    (type_identifier) @identifier
    (identifier) @identifier
  ]
  (class_body
    ("{") @child.first
  )
) @definition.class @root

(enum_declaration
  (identifier) @identifier
  (enum_body
    ("{") @child.first
  )
) @definition.class @root

(abstract_class_declaration
  [
    (type_identifier) @identifier
    (identifier) @identifier
  ]
  (class_body
    ("{") @child.first
  )
) @definition.class @root

(interface_declaration
  (type_identifier) @identifier
  (object_type
    ("{") @child.first
  )
) @definition.class @root

(abstract_method_signature
  (property_identifier) @identifier
) @definition.function @root
