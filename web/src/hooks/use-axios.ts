import axios, { type CreateAxiosDefaults } from "axios"
import { useMemo } from "react";

const defaultClient: CreateAxiosDefaults = {
    baseURL: import.meta.env.VITE_API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
};

const useAxios = () => {
    const client = useMemo(() => {
        return axios.create(defaultClient);
    }, [])

    return client;
}

export default useAxios;