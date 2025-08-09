import { create } from "zustand";

export interface SettingState {
    showMarker: boolean;
    showLabel: boolean;
    setShowMarker: (showMarker: boolean) => void;
    setShowLabel: (showLabel: boolean) => void;
}

const useSettings = create<SettingState>((set) => ({
    showMarker: true,
    showLabel: true,
    setShowMarker: (showMarker: boolean) => set(() => ({ showMarker })),
    setShowLabel: (showLabel: boolean) => set(() => ({ showLabel })),
}));

export default useSettings;