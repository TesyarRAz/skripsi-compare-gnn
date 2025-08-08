import {type AxiosInstance} from "axios"

export interface PredictTSPRequest {
    model: string;
    coords: {   
        lat: number;
        lon: number;
    }[]
}

export interface PredictTSPResponse {
    cost: number
    tour_mask: number[]
}

export const predictTSP = async (client: AxiosInstance, request: PredictTSPRequest): Promise<PredictTSPResponse> => {
    const response = await client.post('/predict/coord', request);
    
    if (response.status != 200) {
        throw new Error(response.statusText);
    }

    return response.data;
}