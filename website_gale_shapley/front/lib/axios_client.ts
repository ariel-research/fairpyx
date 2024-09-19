import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

// Create an Axios instance with default configuration
const apiClient: AxiosInstance = axios.create({
    baseURL: "http://localhost:50010/api", // Replace with your backend URL
    timeout: 10000, // Optional: adjust the timeout as needed
    headers: {
        'Content-Type': 'application/json',
        // Add any other default headers you need
    },
});

// Define the shape of your error response if you know it
interface ErrorResponse {
    detail: string;
    [key: string]: any; // To accommodate any additional properties
}

// Handle responses
const handleResponse = <T>(response: AxiosResponse<T>): T => {
    return response.data;
};

// Handle errors
const handleError = (error: AxiosError): never => {
    if (error.response) {
        // Server responded with a status other than 2xx
        const data = error.response.data as ErrorResponse; // Type assertion
        const message = data.detail || 'Server Error';
        console.error('Server Error:', data);
        throw new AxiosError(message);
    } else if (error.request) {
        // Request was made but no response received
        console.error('Network Error:', error.request);
        throw new Error('Network Error');
    } else {
        // Something happened in setting up the request
        console.error('Error:', error.message);
        throw new Error(error.message);
    }
};

// Define API methods

export const get = async <T>(url: string, params?: Record<string, unknown>, headers?: Record<string, string>): Promise<T> => {
    try {
        const config = {
            params,
            headers
        };
        const response: AxiosResponse<T> = await apiClient.get(url, config);
        return handleResponse(response);
    } catch (error) {
        return handleError(error as AxiosError);
    }
};

export const post = async <T>(url: string, data: Record<string, unknown>, headers?: Record<string, string>): Promise<T> => {
    try {
        const config = {
            headers
        };
        const response: AxiosResponse<T> = await apiClient.post(url, data, config);
        return handleResponse(response);
    } catch (error) {
        return handleError(error as AxiosError);
    }
};

export const put = async <T>(url: string, data: Record<string, unknown>, headers?: Record<string, string>): Promise<T> => {
    try {
        const config = {
            headers
        };
        const response: AxiosResponse<T> = await apiClient.put(url, data, config);
        return handleResponse(response);
    } catch (error) {
        return handleError(error as AxiosError);
    }
};

export const del = async <T>(url: string, headers?: Record<string, string>): Promise<T> => {
    try {
        const config = {
            headers
        };
        const response: AxiosResponse<T> = await apiClient.delete(url, config);
        return handleResponse(response);
    } catch (error) {
        return handleError(error as AxiosError);
    }
};
