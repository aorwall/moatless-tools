"use client"

import { useState } from "react"
import { Info } from "lucide-react"
import { Switch } from "@/lib/components/ui/switch"
import { Input } from "@/lib/components/ui/input"
import { Label } from "@/lib/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/lib/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/lib/components/ui/card"
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/lib/components/ui/tooltip"

interface SelectorConfig {
  type: string
  properties?: Record<string, any>
}

interface ValueFunctionConfig {
  type: string
  properties?: Record<string, any>
}

interface FeedbackConfig {
  type: string
  maxTrajectoryLength?: number
  includeParentInfo?: boolean
  includeTree?: boolean
  includeNodeSuggestion?: boolean
}

export default function DynamicSettingsForm() {
  const [selector, setSelector] = useState<SelectorConfig>({ type: "SimpleSelector" })
  const [valueFunction, setValueFunction] = useState<ValueFunctionConfig>({ type: "SwebenchValueFunction" })
  const [feedback, setFeedback] = useState<FeedbackConfig>({
    type: "FeedbackAgent",
    maxTrajectoryLength: 30,
    includeParentInfo: false,
    includeTree: false,
    includeNodeSuggestion: false,
  })

  return (
    <TooltipProvider>
      <div className="space-y-6">
        {/* Selector Section */}
        <Card>
          <CardHeader>
            <CardTitle>Selector</CardTitle>
            <CardDescription>Component that selects which nodes to expand</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select value={selector.type} onValueChange={(value) => setSelector({ type: value })}>
              <SelectTrigger>
                <SelectValue placeholder="Select type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="SimpleSelector">SimpleSelector</SelectItem>
                <SelectItem value="RandomSelector">RandomSelector</SelectItem>
                <SelectItem value="PrioritySelector">PrioritySelector</SelectItem>
              </SelectContent>
            </Select>

            <div className="text-sm font-medium text-muted-foreground">Selected: {selector.type}</div>

            {/* Conditional properties based on selector type */}
            {selector.type === "PrioritySelector" && (
              <div className="space-y-4 pt-4 border-t">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label>Priority Threshold</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>Sets the threshold for node selection priority</TooltipContent>
                    </Tooltip>
                  </div>
                  <Input type="number" min={0} max={1} step={0.1} defaultValue={0.5} />
                </div>
              </div>
            )}

            {selector.type === "RandomSelector" && (
              <div className="space-y-4 pt-4 border-t">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label>Random Seed</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>Seed for random number generation</TooltipContent>
                    </Tooltip>
                  </div>
                  <Input type="number" defaultValue={42} />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Value Function Section */}
        <Card>
          <CardHeader>
            <CardTitle>Value Function</CardTitle>
            <CardDescription>Component that evaluates node quality</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select value={valueFunction.type} onValueChange={(value) => setValueFunction({ type: value })}>
              <SelectTrigger>
                <SelectValue placeholder="Select type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="SwebenchValueFunction">SwebenchValueFunction</SelectItem>
                <SelectItem value="CustomValueFunction">CustomValueFunction</SelectItem>
              </SelectContent>
            </Select>

            <div className="text-sm font-medium text-muted-foreground">Selected: {valueFunction.type}</div>

            {/* Conditional properties based on value function type */}
            {valueFunction.type === "CustomValueFunction" && (
              <div className="space-y-4 pt-4 border-t">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label>Evaluation Method</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>Method used to evaluate nodes</TooltipContent>
                    </Tooltip>
                  </div>
                  <Select defaultValue="weighted">
                    <SelectTrigger>
                      <SelectValue placeholder="Select method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="weighted">Weighted Average</SelectItem>
                      <SelectItem value="neural">Neural Network</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}

            {valueFunction.type === "SwebenchValueFunction" && (
              <div className="space-y-4 pt-4 border-t">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label>Benchmark Dataset</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>Dataset used for benchmarking</TooltipContent>
                    </Tooltip>
                  </div>
                  <Select defaultValue="standard">
                    <SelectTrigger>
                      <SelectValue placeholder="Select dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="standard">Standard</SelectItem>
                      <SelectItem value="extended">Extended</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Feedback Generator Section */}
        <Card>
          <CardHeader>
            <CardTitle>Feedback Generator</CardTitle>
            <CardDescription>Component that generates feedback for nodes</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <Select value={feedback.type} onValueChange={(value) => setFeedback({ ...feedback, type: value })}>
              <SelectTrigger>
                <SelectValue placeholder="Select type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="FeedbackAgent">FeedbackAgent</SelectItem>
                <SelectItem value="CustomFeedback">CustomFeedback</SelectItem>
              </SelectContent>
            </Select>

            <div className="text-sm font-medium text-muted-foreground">Selected: {feedback.type}</div>

            {feedback.type === "FeedbackAgent" && (
              <div className="space-y-6 pt-4 border-t">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="maxTrajectoryLength">Max Trajectory Length</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>Maximum length of trajectory</TooltipContent>
                    </Tooltip>
                  </div>
                  <Input
                    id="maxTrajectoryLength"
                    type="number"
                    value={feedback.maxTrajectoryLength}
                    onChange={(e) =>
                      setFeedback({
                        ...feedback,
                        maxTrajectoryLength: Number.parseInt(e.target.value),
                      })
                    }
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Include Parent Info</Label>
                      <p className="text-sm text-muted-foreground">Include information about parent nodes</p>
                    </div>
                    <Switch
                      checked={feedback.includeParentInfo}
                      onCheckedChange={(checked) =>
                        setFeedback({
                          ...feedback,
                          includeParentInfo: checked,
                        })
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Include Tree</Label>
                      <p className="text-sm text-muted-foreground">Include tree structure in feedback</p>
                    </div>
                    <Switch
                      checked={feedback.includeTree}
                      onCheckedChange={(checked) =>
                        setFeedback({
                          ...feedback,
                          includeTree: checked,
                        })
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Include Node Suggestion</Label>
                      <p className="text-sm text-muted-foreground">Include suggestions for node improvements</p>
                    </div>
                    <Switch
                      checked={feedback.includeNodeSuggestion}
                      onCheckedChange={(checked) =>
                        setFeedback({
                          ...feedback,
                          includeNodeSuggestion: checked,
                        })
                      }
                    />
                  </div>
                </div>
              </div>
            )}

            {feedback.type === "CustomFeedback" && (
              <div className="space-y-4 pt-4 border-t">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label>Feedback Model</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>Model used to generate feedback</TooltipContent>
                    </Tooltip>
                  </div>
                  <Select defaultValue="gpt4">
                    <SelectTrigger>
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gpt4">GPT-4</SelectItem>
                      <SelectItem value="claude">Claude</SelectItem>
                      <SelectItem value="custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  )
}

