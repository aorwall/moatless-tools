import { Outlet } from 'react-router-dom';

export function SettingsLayout() {
  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>
    </div>
  );
} 