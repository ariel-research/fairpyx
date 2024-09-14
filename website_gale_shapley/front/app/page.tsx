import AllocationForm from "@/components/AllocationForm";
import Link from "next/link";

export default async function Home() {
  return (
    <div className="flex flex-col items-center w-full h-full space-y-5 pt-5 pl-2 pr-2">
      <p className="text-lg md:text-3xl font-bold">
        Course allocation: Gale-Shapley protocol
      </p>
      <span className="w-full md:w-2/3">
        This algorithm is based on the following paper:
        <br />
        <Link className="underline hover:text-blue-500" href="https://doi.org/10.1111/j.1468-2354.2009.00572.x">"Course bidding at business schools", by Tayfun Sönmez and M. Utku Ünver (2010)</Link>
        <br />
        Programmed by: <Link className="underline hover:text-blue-500" href="https://github.com/zachibenshitrit">Zachi Ben Shitrit</Link>
        <br />
        <br />
        This form allows you to allocate courses to students based on the Gale-Shapley algorithm.
        Please provide the following data:
        <ul className="list-disc list-inside mt-2">
          <li><strong className="underline">Course Capabilities:</strong> Enter a JSON string representing the capacities of each course.</li>
          <li><strong className="underline">Agent Capabilities:</strong> Enter a JSON string representing the capacities of each agent.</li>
          <li><strong className="underline">Bids:</strong> Enter a JSON string representing the bids for each agent on courses.</li>
          <li><strong className="underline">Course Order Per Student:</strong> Enter a JSON string representing the course order preferences for each student.</li>
          <li><strong className="underline">Tie-Breaking Lottery:</strong> Enter a JSON string representing the tie-breaking values for each student.</li>
        </ul>
        After filling out the form, the Gale-Shapley algorithm will process the data to allocate courses to students based on their preferences and the provided constraints.
      </span>
      <AllocationForm className="w-full md:w-2/3" />
    </div>
  );
}
