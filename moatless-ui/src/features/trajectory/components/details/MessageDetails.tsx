interface MessageDetailsProps {
  content: {
    message: string;
  };
  type: "user_message" | "assistant_message" | "thought";
}

export const MessageDetails = ({ content, type }: MessageDetailsProps) => {
  return (
    <div className="space-y-2">
      <div className="text-sm text-gray-500">
        {type === "user_message"
          ? "User Message"
          : type === "assistant_message"
            ? "Assistant Message"
            : "Thought"}
      </div>
      <div className="prose prose-sm max-w-none">
        <pre className="whitespace-pre-wrap rounded-lg bg-gray-50 p-4 text-sm text-gray-700">
          {content.message}
        </pre>
      </div>
    </div>
  );
};
