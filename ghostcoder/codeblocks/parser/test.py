
import tree_sitter_languages
from tree_sitter_languages import get_language

t_query = """(lexical_declaration
  (variable_declarator
    (identifier) @identifier
    (arrow_function
      (formal_parameters) @block_type
      (statement_block) @first_child
    )
  )
)
"""

content1 = """const[sortBy, setSortBy] = useState <SortBy> ();"""
content2 = """
export const CareUnitSelector = ({
  careUnits,
  listedCareUnit,
  isOnline = false,
  initialSelection,
  isSearchAndSortingEnabled,
  onCareUnitSelected,
  allowSortingByAvailability = true,
  ...containerAttributes
}: CareUnitSelectorProps) => {
  const lang = useTranslations();
  const [sortBy, setSortBy] = useState<CareUnitsSortBy>(CARE_UNITS_SORT.NAME);
}"""

content = """const Foo = () => {}"""

content_in_bytes = bytes(content, "utf8")
tree_parser = tree_sitter_languages.get_parser("tsx")
language = get_language("tsx")

tree = tree_parser.parse(content_in_bytes)

current_node = tree.root_node.children[0]

query = language.query(t_query)
captures = query.captures(current_node)

print(t_query)
captures = list(captures)

saw = set()
for node, tag in captures:

    print(tag)
    if tag.startswith("name.definition."):
        kind = "def"
    elif tag.startswith("name.reference."):
        kind = "ref"
    else:
        continue

    saw.add(kind)
