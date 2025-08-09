import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { TSPNode } from "../models/tspnode";
import type { LatLngTuple } from "leaflet";

export interface NodesState {
    nodes: TSPNode[];
    routes: LatLngTuple[];
    update: (updated: TSPNode) => void;
    add: (center: LatLngTuple) => void;
    remove: (id: string) => void;
    setRoutes: (routes: LatLngTuple[]) => void;
    clearNodes: () => void; // Optional method to clear nodes
}

const useNodes = create<NodesState>()(
    persist(
        (set) => ({
            nodes: [],
            routes: [],
            update(updated) {
                set((state) => ({
                    nodes: state.nodes.map((node) =>
                        node.id === updated.id ? { ...node, ...updated } : node
                    ),
                    routes: [],
                }));
            },
            add(center) {
                set((state) => ({
                    nodes: [
                        ...state.nodes,
                        {
                            lat: center[0] + Math.random() * 0.01,
                            lng: center[1] + Math.random() * 0.01,
                            id: Date.now().toString(),
                            name: `Node ${state.nodes.length + 1}`,
                        },
                    ],
                    routes: [],
                }));
            },
            remove(id) {
                set((state) => ({
                    nodes: state.nodes.filter((node) => node.id !== id),
                    routes: [],
                }));
            },
            clearNodes() {
                set(() => ({
                    nodes: [],
                    routes: [],
                }));
            },
            setRoutes(routes) {
                set(() => ({
                    routes: routes,
                }));
            },
        }),
        {
            name: "nodes-storage", // key di localStorage
            partialize: (state) => ({ nodes: state.nodes }), // cuma save nodes, routes nggak
        }
    )
);

export default useNodes;
