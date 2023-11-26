(jsx_element
  (jsx_opening_element
    (identifier) @identifier
  )
  (_) @child.first
  (jsx_closing_element) @child.last
) @definition.block @root