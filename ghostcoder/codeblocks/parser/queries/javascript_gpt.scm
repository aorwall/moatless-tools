(program . [
  ;; corner case queries to handle incorrect responses from the LLM
  (
    (expression_statement
      (call_expression
        [
          (identifier) @identifier
          (member_expression
            (identifier) @identifier)
        ]
        (arguments)
      )
    )
    (statement_block
      ("{") @child.first
      ("}") @child.last  @definition.function
    )
  )
]) @root

;; Sometimes GPT just returns "..." when commenting out code
((ERROR) @error
  (#eq? @error "...")) @root @definition.comment