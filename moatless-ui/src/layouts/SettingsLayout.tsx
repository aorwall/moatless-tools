import { Outlet } from "react-router-dom";

export function SettingsLayout() {
  return (
    <div className="flex h-full min-h-0">
      <div className="flex-1 min-h-0">
        <Outlet />
      </div>
    </div>
  );
}
