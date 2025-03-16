import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { useResumeTrajectory } from "@/lib/hooks/useResumeTrajectory";
import { Node, TimelineItem, Trajectory } from "@/lib/types/trajectory";
import { cn } from "@/lib/utils";
import { Bot, MessageSquare } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ChatInput, ChatInputRef } from "./ChatInput";
import { ActionChatItem } from "./items/ActionChatItem";
import { ArtifactChatItem } from "./items/ArtifactChatItem";
import { MessageChatItem } from "./items/MessageChatItem";
import { ThoughtChatItem } from "./items/ThoughtChatItem";
import { ChatMessage, ChatMessageGroup } from "./types";

interface ChatProps {
  trajectory: Trajectory;
}

export function Chat({ trajectory }: ChatProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<ChatInputRef>(null);
  const [expandedMessages, setExpandedMessages] = useState<Set<string>>(
    new Set(),
  );
  const resumeTrajectory = useResumeTrajectory();

  const disabled =
    trajectory.status === "running" || resumeTrajectory.isPending;

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    const scrollToBottom = () => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    };

    scrollToBottom();

    // ...after a small delay to ensure DOM has updated
    const timeoutId = setTimeout(scrollToBottom, 100);

    return () => clearTimeout(timeoutId);
  }, [trajectory.nodes?.length]);

  if (!trajectory.nodes) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="text-muted-foreground">Loading chat...</span>
      </div>
    );
  }

  // Transform trajectory nodes into chat messages
  const messages: ChatMessage[] = [];
  trajectory.nodes.forEach((node: Node) => {
    // Add items in sequence
    node.items?.forEach((item: TimelineItem) => {
      if (
        [
          "user_message",
          "assistant_message",
          "thought",
          "action",
          "artifact",
        ].includes(item.type)
      ) {
        messages.push({
          ...item,
          nodeId: node.nodeId,
          id: `${node.nodeId}-${item.type}-${messages.length}`,
          trajectoryId: trajectory.trajectory_id,
          timestamp: new Date().toISOString(),
        });
      }
    });
  });

  // Group consecutive messages by sender, keeping artifacts with their related messages
  const messageGroups: ChatMessageGroup[] = messages.reduce(
    (groups: ChatMessageGroup[], message) => {
      const isUser =
        message.type === "user_message" ||
        (message.type === "artifact" &&
          (message.content as any).actor === "user");
      const lastGroup = groups[groups.length - 1];

      // If it's an artifact, add it to the current group if it matches the sender
      if (message.type === "artifact") {
        if (lastGroup && lastGroup.isUser === isUser) {
          lastGroup.messages.push(message);
          return groups;
        }
      }

      // For regular messages, start a new group if sender changes
      if (
        lastGroup &&
        lastGroup.isUser === isUser &&
        message.type !== "artifact"
      ) {
        lastGroup.messages.push(message);
      } else {
        groups.push({ messages: [message], isUser });
      }

      return groups;
    },
    [],
  );

  const toggleMessageExpand = (messageId: string) => {
    setExpandedMessages((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  const onSendMessage = (message: string, agentId: string, modelId: string) => {
    resumeTrajectory.mutate({
      trajectoryId: trajectory.trajectory_id,
      agentId,
      modelId,
      message,
      onSuccess: () => {
        // Clear the input only on success
        chatInputRef.current?.clear();
      },
    });
  };

  return (
    <div className="flex flex-col h-full w-full">
      <ScrollArea ref={scrollRef} className="flex-1 px-4">
        <div className="space-y-6 py-4">
          {messageGroups.map((group, groupIndex) => (
            <div
              key={groupIndex}
              className={cn(
                "flex gap-3",
                group.isUser ? "flex-row-reverse" : "flex-row",
              )}
            >
              <div
                className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center self-start rounded-full",
                  group.isUser ? "bg-primary" : "bg-muted",
                )}
              >
                {group.isUser ? (
                  <MessageSquare className="h-4 w-4" />
                ) : (
                  <Bot className="h-4 w-4" />
                )}
              </div>

              <div className="flex min-w-0 max-w-[80%] flex-col gap-2">
                {group.messages.map((message, messageIndex) => {
                  const messageId = `${message.nodeId}-${messageIndex}`;
                  const isExpanded = expandedMessages.has(messageId);

                  switch (message.type) {
                    case "user_message":
                    case "assistant_message":
                      return (
                        <MessageChatItem
                          key={messageIndex}
                          message={message}
                          isUser={group.isUser}
                          isExpanded={isExpanded}
                          onExpandChange={() => toggleMessageExpand(messageId)}
                        />
                      );
                    case "thought":
                      return (
                        <ThoughtChatItem key={messageIndex} message={message} />
                      );
                    case "action":
                      return (
                        <ActionChatItem key={messageIndex} message={message} />
                      );
                    case "artifact":
                      return (
                        <ArtifactChatItem
                          key={messageIndex}
                          message={message}
                        />
                      );
                    default:
                      return null;
                  }
                })}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
      <ChatInput
        ref={chatInputRef}
        onSend={onSendMessage}
        disabled={disabled}
        agentId={trajectory.agent_id}
        modelId={trajectory.model_id}
      />
    </div>
  );
}
