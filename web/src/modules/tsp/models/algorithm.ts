import _ from "lodash";

export interface Algorithm {
    key : string;
    group: string;
    name: string;
}

export const algorithms: Algorithm[] = [
    { key: "gcn/10_30", group: "GCN", name: "GCN 10-30" },
    { key: "gcn/10_50", group: "GCN", name: "GCN 10-50" },
    { key: "gcn/20_30", group: "GCN", name: "GCN 20-30" },
    { key: "gcn/20_50", group: "GCN", name: "GCN 20-50" },
    { key: "gat/10_30", group: "GAT", name: "GAT 10-30" },
    { key: "gat/10_50", group: "GAT", name: "GAT 10-50" },
    { key: "gat/20_30", group: "GAT", name: "GAT 20-30" },
    { key: "gat/20_50", group: "GAT", name: "GAT 20-50" },
    { key: "gat_v2/10_30", group: "GAT V2", name: "GAT V2 10-30" },
    { key: "gat_v2/10_50", group: "GAT V2", name: "GAT V2 10-50" },
    { key: "gat_v2/20_30", group: "GAT V2", name: "GAT V2 20-30" },
    { key: "gat_v2/20_50", group: "GAT V2", name: "GAT V2 20-50" },
    // { key: "ant_colony", group: "Other Algorithm", name: "Ant Colony" },
    { key: "nearest_neighbor", group: "Other Algorithm", name: "Nearest Neighbor" },
    { key: "held_karp", group: "Other Algorithm", name: "Held Karp" },
];

export const groupedAlgorithms = _.groupBy(algorithms, 'group') as Record<string, Algorithm[]>;