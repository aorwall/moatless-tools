(program . [
  ;; corner case queries to handle incorrect responses from the LLM
  (
    (expression_statement
      (call_expression
        [
          (identifier) @identifier @definition.function
          (member_expression
            (identifier) @identifier @definition.function)
        ]
        (arguments)
      )
    )
    (statement_block
      ("{") @child.first
      ("}") @child.last
    )
  )
]) @root
