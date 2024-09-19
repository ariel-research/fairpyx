import { z } from "zod";

// Reusable JSON validation schema
const jsonStringSchema = z.string().refine((data) => {
    try {
        JSON.parse(data);
        return true;
    } catch {
        return false;
    }
}, {
    message: 'Invalid JSON format',
});

// Form schema using the reusable JSON validation schema
export const formSchema = z.object({
    courseCapabilities: jsonStringSchema,
    agentCapabilities: jsonStringSchema,
    bids: jsonStringSchema,
    courseOrderPerStudent: jsonStringSchema,
    tieBrakingLottery: jsonStringSchema,
});

export interface CoursePreferences {
    [key: string]: string[];
  }
export interface AllocationResult {
  allocation: CoursePreferences;
  logs: string[];
}
