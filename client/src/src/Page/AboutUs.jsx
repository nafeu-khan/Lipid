import React from "react";

function AboutUs() {
  return (
    <div className="bg-[whitesmoke] min-h-screen grid place-items-center">
      <div className="max-w-lg text-center border-2 p-6 shadow">
        <div
          className="flex items-center justify-center mb-6 cursor-pointer"
          onClick={() => (window.location.href = "/")}
        >
          <img src="logo.gif" alt="" className="w-20" />
          <h2 className="font-serif text-xl font-semibold tracking-wider">
            Lipid Membrane
          </h2>
        </div>
        <p className="font-mono text-gray-700/90">
          Lorem ipsum dolor sit amet consectetur adipisicing elit. Praesentium,
          quos tenetur! Debitis eos assumenda autem incidunt officia corrupti
          quis iste perspiciatis earum, reiciendis, nulla, ut quae tempora sequi
          magnam voluptatem culpa amet deserunt laudantium reprehenderit!
          Voluptatem pariatur officiis at quae?
        </p>
        <p className="text-sm mt-4 text-gray-700/80">
          {" "}
          <span className="font-medium text-gray-800 underline mr-1.5">
            Contact us:{" "}
          </span>{" "}
          test@test.com
        </p>
        <p className="text-sm mt-1 text-gray-700/80">
          {" "}
          <span className="font-medium text-gray-800 underline mr-1.5">
            Mobile:{" "}
          </span>{" "}
          +880123456789
        </p>
      </div>
    </div>
  );
}

export default AboutUs;
