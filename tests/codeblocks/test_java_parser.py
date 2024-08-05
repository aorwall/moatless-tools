from moatless.codeblocks import JavaParser


def _verify_parsing(content, assertion, apply_gpt_tweaks=True, debug=True):
    parser = JavaParser(apply_gpt_tweaks=apply_gpt_tweaks, debug=debug)

    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_references=True, show_spans=True, show_tokens=True))

    assert codeblock.to_string() == content

    assertion(codeblock)


def test_override_function():
    content = """public class Foo {
    @Override
    public void helloWorld() {
        // comment
        System.out.println("Hello World!");
    }
}"""

    def assertion(codeblock):
        print(
            codeblock.to_tree(
                include_references=True,
                show_spans=True,
                show_tokens=True,
                only_identifiers=False,
            )
        )

    _verify_parsing(content, assertion)


def test_interface():
    content = """public interface Foo {
    /**
     * Prints "Hello World!" to the console.
     */
    // TODO: Test
    void helloWorld();
}
"""

    def assertion(codeblock):
        print(
            codeblock.to_tree(
                include_references=True, show_spans=True, show_tokens=True
            )
        )

    _verify_parsing(content, assertion)
