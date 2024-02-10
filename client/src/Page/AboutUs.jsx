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
        <p className="font-mono text-gray-700/90"></p>
        {/* <p className="text-sm mt-4 text-gray-700/80">
          {" "}
          <span className="font-medium text-gray-800 underline mr-1.5">
            Contact us:{" "}
          </span>{" "}
        </p> */}
        {/* <p className="text-sm mt-1 text-gray-700/80">
          {" "}
          <span className="font-medium text-gray-800 underline mr-1.5">
            Mobile:{" "}
          </span>{" "}
          
        </p> */}
      </div>
    </div>
  );
}

export default AboutUs;
