import z from "zod";

export const tspNodeSchema = z.object({
    id: z.string(),
    name: z.string(),

    lat: z.number(),
    lng: z.number()
})

export type TSPNode = z.infer<typeof tspNodeSchema>;

export interface Benchmark {
    model: string
    cost: number
}
