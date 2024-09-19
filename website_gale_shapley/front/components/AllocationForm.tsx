"use client";
import clsx from "clsx";
import { Textarea } from "./ui/textarea";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "./ui/form";
import { z } from "zod";
import { Button } from "./ui/button";
import { CoursePreferences, formSchema } from "@/lib/formSchemas";
import { submitForm } from "@/lib/actions";
import { AllocationResult } from "@/lib/formSchemas";
import { AllocationResultDialog } from "./AllocationResultDialog";
import { useState } from "react";

export default function AllocationForm({ className }: { className?: string }) {
    const [result, setResult] = useState<AllocationResult | undefined>();
    const [open, setOpen] = useState(false);

    // Define your form.
    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            courseCapabilities: "",
            agentCapabilities: "",
            bids: "",
            courseOrderPerStudent: "",
            tieBrakingLottery: "",
        },
    });

    // Define a submit handler.
    async function onSubmit(values: z.infer<typeof formSchema>) {
        try {
            const temp = await submitForm(values);
            console.log(temp);
            setResult(temp);
            setOpen(true);
        } catch (error) {
            console.error("Error submitting form:", error);
        }
    }

    // Set example values for the form fields
    function handleDefaultValuesExample() {
        form.setValue("courseCapabilities", '{"Course1": 4, "Course2": 4, "Course3": 2, "Course4": 3, "Course5": 2}');
        form.setValue("agentCapabilities", '{"Alice": 3, "Bob": 3, "Chana": 3, "Dana": 3, "Dor": 3}');
        form.setValue("bids", '{"Alice": {"Course1": 20, "Course2": 15, "Course3": 35, "Course4": 10, "Course5": 20}, "Bob": {"Course1": 30, "Course2": 15, "Course3": 20, "Course4": 20, "Course5": 15}, "Chana": {"Course1": 40, "Course2": 10, "Course3": 25, "Course4": 10, "Course5": 15}, "Dana": {"Course1": 10, "Course2": 10, "Course3": 15, "Course4": 30, "Course5": 35}, "Dor": {"Course1": 25, "Course2": 20, "Course3": 30, "Course4": 10, "Course5": 15}}');
        form.setValue("courseOrderPerStudent", '{"Alice": ["Course5", "Course3", "Course1", "Course2", "Course4"], "Bob": ["Course1", "Course4", "Course5", "Course2", "Course3"], "Chana": ["Course5", "Course1", "Course4", "Course3", "Course2"], "Dana": ["Course3", "Course4", "Course1", "Course5", "Course2"], "Dor": ["Course5", "Course1", "Course4", "Course3", "Course2"]}');
        form.setValue("tieBrakingLottery", '{"Alice": 0.6, "Bob": 0.4, "Chana": 0.3, "Dana": 0.8, "Dor": 0.2}');
    }

    // Clear all form fields
    function handleClearForm() {
        form.setValue("courseCapabilities", "");
        form.setValue("agentCapabilities", "");
        form.setValue("bids", "");
        form.setValue("courseOrderPerStudent", "");
        form.setValue("tieBrakingLottery", "");
    }

    return (
        <div className={clsx("flex flex-col space-y-5 w-full h-full", className)}>
            <div className="flex flex-row justify-between">
                <Button onClick={handleClearForm}>Clear Form</Button>
                <Button onClick={handleDefaultValuesExample}>Fill Example Values</Button>
            </div>

            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                    <FormField
                        control={form.control}
                        name="courseCapabilities"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Courses and Capacities</FormLabel>
                                <FormControl>
                                    <Textarea {...field} placeholder="Enter a JSON representing the capacities of each course." />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={form.control}
                        name="agentCapabilities"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Agents and Capacities</FormLabel>
                                <FormControl>
                                    <Textarea {...field} placeholder="Enter a JSON representing the capacities of each agent." />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={form.control}
                        name="bids"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Bids</FormLabel>
                                <FormControl>
                                    <Textarea {...field} placeholder="Enter a JSON representing the bids for each agent on courses." />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={form.control}
                        name="courseOrderPerStudent"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Course Order Per Student</FormLabel>
                                <FormControl>
                                    <Textarea {...field} placeholder="Enter a JSON representing the course order preferences for each student." />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={form.control}
                        name="tieBrakingLottery"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Tie-breaking Lottery</FormLabel>
                                <FormControl>
                                    <Textarea {...field} placeholder="Enter a JSON representing the tie-breaking values for each student." />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <Button type="submit" className="w-full bg-blue-500 hover:bg-blue-400">Submit</Button>
                </form>
            </Form>

            <AllocationResultDialog result={result} open={open} setOpen={setOpen} />
        </div>
    );
}
