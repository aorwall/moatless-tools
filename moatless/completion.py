def completion():
    response_format = response_format_by_model(self.model)

    if response_format == LLMResponseFormat.ANTHROPIC_TOOLS and self.action_type():
        try:
            tools = []
            if hasattr(self.action_type(), "available_actions"):
                for action in self.action_type().available_actions():
                    tools.append(action.anthropic_schema)
            else:
                tools.append(self.action_type().anthropic_schema)

            if self.model == "anthropic.claude-3-5-sonnet-20240620-v1:0":
                anthropic_client = AnthropicBedrock()
                completion_response = anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=messages[0]["content"],
                    tool_choice={"type": "any"},
                    tools=tools,
                    messages=messages[1:],
                )
            else:
                anthropic_client = Anthropic()

                apply_cache_control(messages[0])
                # apply_cache_control(messages[-1])

                completion_response = (
                    anthropic_client.beta.prompt_caching.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=messages[0]["content"],
                        tool_choice={"type": "any"},
                        tools=tools,
                        messages=messages[1:],
                    )
                )

            completion = Completion.from_llm_completion(
                input_messages=messages,
                completion_response=completion_response,
                model=self.model,
            )

            try:
                action_request = None
                if hasattr(self.action_type(), "available_actions"):
                    for block in completion_response.content:
                        if isinstance(block, ToolUseBlock):
                            action = None
                            for (
                                available_action
                            ) in self.action_type().available_actions():
                                if available_action.__name__ == block.name:
                                    action = available_action
                                    break

                            if not action:
                                raise ValueError(f"Unknown action {block.name}")

                            tool_action_request = action.model_validate(block.input)

                            action_request = self.action_type()(
                                action=tool_action_request
                            )

                            # TODO: We only support one action at the moment
                            break
                        else:
                            logger.warning(f"Unexpected block {block}]")
                else:
                    action_request = self.action_type().from_response(
                        completion_response, mode=instructor.Mode.ANTHROPIC_TOOLS
                    )

                if not action_request:
                    raise ValueError(
                        f"Failed to parse action request from completion response. Completion: {completion_response}"
                    )
            except Exception as e:
                logger.exception(
                    f"Failed to parse action request from completion response. Completion: {completion_response}"
                )
                raise e

            return action_request, completion

        except Exception as e:
            logger.error(f"Failed to get completion response from anthropic: {e}")
            raise e

    if (
        self.action_type() is None
        and self.model.startswith("claude")
        or self.model.startswith("anthropic.claude")
    ):
        if self.model == "anthropic.claude-3-5-sonnet-20240620-v1:0":
            anthropic_client = AnthropicBedrock()
            completion_response = anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=messages[0]["content"],
                messages=messages[1:],
            )
        else:
            anthropic_client = Anthropic()

            apply_cache_control(messages[0])
            # apply_cache_control(messages[-1])

            completion_response = (
                anthropic_client.beta.prompt_caching.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=messages[0]["content"],
                    messages=messages[1:],
                )
            )

        completion = Completion.from_llm_completion(
            input_messages=messages,
            completion_response=completion_response,
            model=self.model,
        )
        action_request = Content(content=completion_response.content[0].text)

        return action_request, completion

    if self.action_type() is None:
        completion_response = litellm.completion(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_words(),
            metadata=metadata,
            messages=messages,
        )
        action_request = Content(
            content=completion_response.choices[0].message.content
        )

        completion = Completion.from_llm_completion(
            input_messages=messages,
            completion_response=completion_response,
            model=self.model,
        )
        return action_request, completion
    elif response_format == LLMResponseFormat.STRUCTURED_OUTPUT:
        client = OpenAI()

        tools = []
        if hasattr(self.action_type(), "available_actions"):
            for action in self.action_type().available_actions():
                tools.append(openai.pydantic_function_tool(action))
        else:
            tools.append(openai.pydantic_function_tool(self.action_type()))

        completion_response = client.beta.chat.completions.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_words(),
            messages=messages,
            tool_choice="required",
            tools=tools,
        )

        tool_call = completion_response.choices[0].message.tool_calls[0]
        if hasattr(self.action_type(), "available_actions"):
            tool_action_request = tool_call.function.parsed_arguments
            action_request = self.action_type()(action=tool_action_request)
        else:
            action_request = tool_call.function.parsed_arguments

        if not action_request:
            raise ValueError(
                f"Failed to parse action request from completion response. Completion: {completion_response}"
            )

        completion = Completion.from_llm_completion(
            input_messages=messages,
            completion_response=completion_response,
            model=self.model,
        )

        return action_request, completion

    elif response_format == LLMResponseFormat.TOOLS:
        tools = []
        if hasattr(self.action_type(), "available_actions"):
            for action in self.action_type().available_actions():
                tools.append(action.openai_tool_schema)
        else:
            tools.append(self.action_type().openai_tool_schema)

        completion_response = litellm.completion(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop_words(),
            tools=tools,
            metadata=metadata,
            messages=messages,
        )

        try:
            action_request = None
            # TODO: We only support one action at the moment
            tool_call = completion_response.choices[0].message.tool_calls[0]

            if hasattr(self.action_type(), "available_actions"):
                action = None
                for available_action in self.action_type().available_actions():
                    if available_action.__name__ == tool_call.function.name:
                        action = available_action
                        break

                if not action:
                    raise ValueError(f"Unknown action {tool_call.function.name}")

                tool_action_request = action.model_validate_json(
                    tool_call.function.arguments
                )
                action_request = self.action_type()(action=tool_action_request)
            else:
                action_request = self.action_type().model_validate_json(
                    tool_call.function.arguments
                )

            if not action_request:
                raise ValueError(
                    f"Failed to parse action request from completion response. Completion: {completion_response}"
                )
        except Exception as e:
            logger.exception(
                f"Failed to parse action request from completion response. Completion: {completion_response}"
            )
            raise e

        if not action_request:
            raise ValueError(
                f"Failed to parse action request from completion response. Completion: {completion_response}"
            )

        completion = Completion.from_llm_completion(
            input_messages=messages,
            completion_response=completion_response,
            model=self.model,
        )

        return action_request, completion
    else:
        client = instructor.from_litellm(
            litellm.completion, mode=instructor.Mode.JSON, metadata=metadata
        )

        try:
            action_request, completion_response = (
                client.chat.completions.create_with_completion(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_words(),
                    response_model=self.action_type(),
                    metadata=metadata,
                    messages=messages,
                )
            )
        except Exception as e:
            logger.error(f"Failed to get completion response from litellm: {e}")
            raise e

        completion = Completion.from_llm_completion(
            input_messages=messages,
            completion_response=completion_response,
            model=self.model,
        )

        return action_request, completion