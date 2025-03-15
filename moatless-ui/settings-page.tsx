import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { SettingsForm } from "@/components/settings-form"
import type { SettingsSchema, SettingsValues } from "@/types/settings"

// Example schema - in a real app, this would likely come from an API
const flowConfigSchema: SettingsSchema = {
  id: "flow-config",
  title: "Flow Configuration",
  sections: [
    {
      id: "basic-info",
      title: "Basic Information",
      description: "Configure the basic settings for your flow",
      fields: [
        {
          id: "flowId",
          type: "text",
          label: "Flow ID",
          defaultValue: "claude_swebench_eval",
          required: true,
        },
        {
          id: "description",
          type: "textarea",
          label: "Description",
          defaultValue: "Coding flow using Claude 3.7 Sonnet with a evaluation value function",
          placeholder: "A brief description of what this flow does",
        },
        {
          id: "flowType",
          type: "select",
          label: "Flow Type",
          tooltip: "The type of flow execution strategy",
          options: [
            { value: "tree", label: "Tree" },
            { value: "linear", label: "Linear" },
          ],
          defaultValue: "tree",
        },
        {
          id: "agent",
          type: "select",
          label: "Agent",
          tooltip: "The agent to use for this flow",
          options: [{ value: "claude", label: "code_and_test_claude_sonnet" }],
          defaultValue: "claude",
        },
      ],
    },
    {
      id: "execution-params",
      title: "Execution Parameters",
      description: "Configure how your flow executes",
      fields: [
        {
          id: "maxIterations",
          type: "number",
          label: "Max Iterations",
          tooltip: "Maximum number of iterations",
          defaultValue: 50,
          min: 1,
        },
        {
          id: "maxCost",
          type: "number",
          label: "Max Cost ($)",
          tooltip: "Maximum cost allowed for the flow in USD",
          defaultValue: 1,
          min: 0,
        },
        {
          id: "maxExpansions",
          type: "number",
          label: "Max Expansions",
          tooltip: "Maximum number of expansions per iteration",
          defaultValue: 1,
          min: 1,
        },
        {
          id: "maxDepth",
          type: "number",
          label: "Max Depth",
          tooltip: "Maximum depth of the flow tree",
          defaultValue: 50,
          min: 1,
        },
        {
          id: "minFinishedNodes",
          type: "number",
          label: "Min Finished Nodes",
          tooltip: "Minimum number of finished nodes required",
          defaultValue: 1,
          min: 0,
        },
        {
          id: "maxFinishedNodes",
          type: "number",
          label: "Max Finished Nodes",
          tooltip: "Maximum number of finished nodes allowed",
          defaultValue: 1,
          min: 1,
        },
      ],
    },
    {
      id: "selector",
      title: "Selector",
      description: "Component that selects which nodes to expand",
      fields: [
        {
          id: "selector",
          type: "component-select",
          label: "Selector Type",
          options: [
            { value: "SimpleSelector", label: "SimpleSelector" },
            { value: "RandomSelector", label: "RandomSelector" },
            { value: "PrioritySelector", label: "PrioritySelector" },
          ],
          defaultValue: "SimpleSelector",
          conditionalFields: {
            PrioritySelector: [
              {
                id: "priorityThreshold",
                type: "number",
                label: "Priority Threshold",
                tooltip: "Sets the threshold for node selection priority",
                defaultValue: 0.5,
                min: 0,
                max: 1,
                step: 0.1,
              },
            ],
            RandomSelector: [
              {
                id: "randomSeed",
                type: "number",
                label: "Random Seed",
                tooltip: "Seed for random number generation",
                defaultValue: 42,
              },
            ],
          },
        },
      ],
    },
    {
      id: "value-function",
      title: "Value Function",
      description: "Component that evaluates node quality",
      fields: [
        {
          id: "valueFunction",
          type: "component-select",
          label: "Value Function Type",
          options: [
            { value: "SwebenchValueFunction", label: "SwebenchValueFunction" },
            { value: "CustomValueFunction", label: "CustomValueFunction" },
          ],
          defaultValue: "SwebenchValueFunction",
          conditionalFields: {
            CustomValueFunction: [
              {
                id: "evaluationMethod",
                type: "select",
                label: "Evaluation Method",
                tooltip: "Method used to evaluate nodes",
                options: [
                  { value: "weighted", label: "Weighted Average" },
                  { value: "neural", label: "Neural Network" },
                ],
                defaultValue: "weighted",
              },
            ],
            SwebenchValueFunction: [
              {
                id: "benchmarkDataset",
                type: "select",
                label: "Benchmark Dataset",
                tooltip: "Dataset used for benchmarking",
                options: [
                  { value: "standard", label: "Standard" },
                  { value: "extended", label: "Extended" },
                ],
                defaultValue: "standard",
              },
            ],
          },
        },
      ],
    },
    {
      id: "feedback-generator",
      title: "Feedback Generator",
      description: "Component that generates feedback for nodes",
      fields: [
        {
          id: "feedbackGenerator",
          type: "component-select",
          label: "Feedback Generator Type",
          options: [
            { value: "FeedbackAgent", label: "FeedbackAgent" },
            { value: "CustomFeedback", label: "CustomFeedback" },
          ],
          defaultValue: "FeedbackAgent",
          conditionalFields: {
            FeedbackAgent: [
              {
                id: "maxTrajectoryLength",
                type: "number",
                label: "Max Trajectory Length",
                tooltip: "Maximum length of trajectory",
                defaultValue: 30,
                min: 1,
              },
              {
                id: "includeParentInfo",
                type: "toggle",
                label: "Include Parent Info",
                description: "Include information about parent nodes",
                defaultValue: false,
              },
              {
                id: "includeTree",
                type: "toggle",
                label: "Include Tree",
                description: "Include tree structure in feedback",
                defaultValue: false,
              },
              {
                id: "includeNodeSuggestion",
                type: "toggle",
                label: "Include Node Suggestion",
                description: "Include suggestions for node improvements",
                defaultValue: false,
              },
            ],
            CustomFeedback: [
              {
                id: "feedbackModel",
                type: "select",
                label: "Feedback Model",
                tooltip: "Model used to generate feedback",
                options: [
                  { value: "gpt4", label: "GPT-4" },
                  { value: "claude", label: "Claude" },
                  { value: "custom", label: "Custom" },
                ],
                defaultValue: "gpt4",
              },
            ],
          },
        },
      ],
    },
    {
      id: "actions",
      title: "Actions",
      description: "Configure the actions for your flow",
      fields: [
        {
          id: "selectedActions",
          type: "dynamic-item-list",
          label: "Selected Actions",
          addButtonText: "Add Action", // Custom button text
          availableItems: [
            {
              id: "appendString",
              name: "AppendString",
              description: "Action to append text content strictly to the end of a file.",
              fields: [
                {
                  id: "autoRunTests",
                  type: "toggle",
                  label: "Auto Run Tests",
                  description: "Whether to automatically run tests after modifying code",
                  defaultValue: false,
                },
                {
                  id: "content",
                  type: "expandable-textarea", // New field type
                  label: "Content to Append",
                  placeholder: "Enter the content to append to the file",
                  minRows: 3,
                  maxRows: 20,
                },
              ],
            },
            {
              id: "createFile",
              name: "CreateFile",
              description: "Action to create a new file with specified content.",
              fields: [
                {
                  id: "autoRunTests",
                  type: "toggle",
                  label: "Auto Run Tests",
                  description: "Whether to automatically run tests after modifying code",
                  defaultValue: false,
                },
                {
                  id: "filePath",
                  type: "text",
                  label: "File Path",
                  placeholder: "Enter the file path",
                  defaultValue: "",
                },
                {
                  id: "content",
                  type: "expandable-textarea",
                  label: "File Content",
                  placeholder: "Enter the content for the new file",
                  minRows: 3,
                  maxRows: 20,
                },
              ],
            },
            {
              id: "findClass",
              name: "FindClass",
              description: "Find a class in the codebase",
              fields: [
                {
                  id: "maxSearchTokens",
                  type: "number",
                  label: "Max Search Tokens",
                  description: "The maximum number of tokens allowed in the search results",
                  defaultValue: 2000,
                },
                {
                  id: "maxIdentifyTokens",
                  type: "number",
                  label: "Max Identify Tokens",
                  description: "The maximum number of tokens allowed in the identified code sections",
                  defaultValue: 8000,
                },
                {
                  id: "maxIdentifyPromptTokens",
                  type: "number",
                  label: "Max Identify Prompt Tokens",
                  description: "The maximum number of tokens allowed in the identify prompt",
                  defaultValue: 16000,
                },
                {
                  id: "maxHits",
                  type: "number",
                  label: "Max Hits",
                  description: "The maximum number of search hits to display",
                  defaultValue: 10,
                },
              ],
            },
            {
              id: "findCodeSnippet",
              name: "FindCodeSnippet",
              description: "Find a code snippet in the codebase",
              fields: [
                {
                  id: "maxSearchTokens",
                  type: "number",
                  label: "Max Search Tokens",
                  description: "The maximum number of tokens allowed in the search results",
                  defaultValue: 2000,
                },
                {
                  id: "maxIdentifyTokens",
                  type: "number",
                  label: "Max Identify Tokens",
                  description: "The maximum number of tokens allowed in the identified code sections",
                  defaultValue: 8000,
                },
                {
                  id: "searchQuery",
                  type: "expandable-textarea",
                  label: "Search Query",
                  placeholder: "Enter your search query",
                  minRows: 2,
                  maxRows: 10,
                },
              ],
            },
          ],
        },
      ],
    },
  ],
}

export default function SettingsPage() {
  const [initialValues, setInitialValues] = useState<SettingsValues>({
    // Default values could be loaded from an API
    flowId: "claude_swebench_eval",
    description: "Coding flow using Claude 3.7 Sonnet with a evaluation value function",
    flowType: "tree",
    agent: "claude",
    maxIterations: 50,
    maxCost: 1,
    maxExpansions: 1,
    maxDepth: 50,
    minFinishedNodes: 1,
    maxFinishedNodes: 1,
    selector: { type: "SimpleSelector" },
    valueFunction: { type: "SwebenchValueFunction" },
    feedbackGenerator: {
      type: "FeedbackAgent",
      maxTrajectoryLength: 30,
      includeParentInfo: false,
      includeTree: false,
      includeNodeSuggestion: false,
    },
    selectedActions: [],
  })

  const handleSave = (values: SettingsValues) => {
    console.log("Saving settings...", values)
    // Here you would typically send this data to your API
    setInitialValues(values) // Update initial values to reflect saved state
  }

  return (
    <TooltipProvider>
      <div className="flex min-h-screen">
        {/* Sidebar */}
        <div className="w-64 border-r bg-muted/30 p-4">
          <div className="space-y-4">
            <Input type="search" placeholder="Search flows..." className="w-full" />
            <nav className="space-y-2">
              <Button variant="ghost" className="w-full justify-start">
                <div className="flex flex-col items-start">
                  <span className="text-sm font-medium">claude_swebench_eval</span>
                  <span className="text-xs text-muted-foreground">Claude 3.7 Sonnet evaluation</span>
                </div>
              </Button>
              {/* Add more flow buttons here */}
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6">
          <div className="mx-auto max-w-4xl">
            <SettingsForm schema={flowConfigSchema} initialValues={initialValues} onSave={handleSave} />
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}

