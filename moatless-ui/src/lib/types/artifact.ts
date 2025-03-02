import type { ContentStructure } from "./content";

export interface Artifact {
  id: string;
  type: string;
  name: string | null;
  created_at: string;
  references: ArtifactReference[];
  status: "updated" | "persisted" | "new" | "unchanged";
  can_persist: boolean;
  data: Record<string, any>;
  content?: ContentStructure;
}
