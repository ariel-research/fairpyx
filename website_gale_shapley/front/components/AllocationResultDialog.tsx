"use client";
import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import { CoursePreferences } from "@/lib/formSchemas";
import { ReactNode, useState } from "react";
import { AllocationResult } from "@/lib/formSchemas";

export function AllocationResultDialog({ result, open, setOpen, children }: { result?: AllocationResult, open: boolean, setOpen: (value: boolean) => void, children?: ReactNode }) {

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
                {children}
            </DialogTrigger>
            <DialogContent className="sm:max-w-auto max-h-[80vh] overflow-auto rounded-lg">
                <DialogHeader>
                    <DialogTitle>Summary</DialogTitle>
                    <DialogDescription>
                        Here are the allocation results:
                    </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                    {result && result.allocation && Object.entries(result.allocation).map(([name, courses]) => (
                        <div key={name}>
                            <h3 className="text-lg font-bold underline">{name}</h3>
                            <ul className="list-disc pl-5">
                                {courses.map((course, index) => (
                                    <li key={index}>{course}</li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
                {result && result.logs && (
                    <div className="mt-6">
                        <h3 className="text-lg font-bold">Algorithm Logs:</h3>
                        <pre className="bg-gray-100 p-2 rounded-md text-sm overflow-x-auto">
                            {result.logs.join('\n')}
                        </pre>
                    </div>
                )}
            </DialogContent>
        </Dialog>
    );
}
