'use server';

import { z } from "zod";
import { post } from "./axios_client";
import { AllocationResult, formSchema } from "./formSchemas";

export async function submitForm(values: z.infer<typeof formSchema>): Promise<AllocationResult> {
    return await post<AllocationResult>(`/divide/`, values);
}