import { ArrowRight } from "@mui/icons-material";
import { CircularProgress, Switch } from "@mui/material";
import React, { useState } from "react";
import ChartComponent from "./ChartComponent";
import MultipleSelectDropdown from "./Components/MultipleSelectDropdown";
import { graph_data } from "./utility";

const data = JSON.parse(graph_data);

function App() {
  const lipid_name = Object.keys(data);
  const [active_lipid, setActiveLipid] = useState([]);
  const [collapse, setCollapse] = useState(false);
  const [image, setImage] = useState();
  const [showPlot, setShowPlot] = useState(false);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState();

  const handleSelectedColumns = (selectedLipids) => {
    const updatedActiveLipids = selectedLipids.map((lipidName) => {
      // Find the lipid object in the current state or create a new one
      const existingLipid = active_lipid.find((l) => l.name === lipidName) || {
        active: false,
        name: lipidName,
        value: 0,
      };
      return existingLipid;
    });
    setActiveLipid(updatedActiveLipids);
  };

  const handleCalculate = async () => {
    setPrediction(null);
    if (!active_lipid || active_lipid.length === 0) return;
    setLoading(true);

    let body = {
      issingle: active_lipid.length,
      lipid_name: active_lipid[0].name,
      percentage: active_lipid[0].value,
    };

    if (active_lipid.length > 1) {
      body = {
        ...body,
        lipid_name2: active_lipid[1].name,
        percentage2: active_lipid[1].value,
      };
    }
    const res = await fetch("http://localhost:8000/prediction/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    console.log(data);
    const imageUrl = `data:image/png;base64,${data.graph}`;
    setImage(imageUrl);
    setPrediction(JSON.stringify(data.pred));
    setLoading(false);
  };

  return (
    <div className="h-screen flex relative overflow-hidden">
      <div
        className={`h-full relative w-[310px] border-r-2 shadow-xl p-4 bg-[whitesmoke] ${
          collapse && "max-w-[0] !p-0"
        }`}
      >
        <span
          className={`absolute right-0 top-1/2 translate-x-1/2 bg-white shadow-lg border-2 cursor-pointer ${
            collapse && "!translate-x-full z-10"
          }`}
          onClick={() => setCollapse(!collapse)}
        >
          <ArrowRight />
        </span>
        {!collapse && (
          <>
            <div className="h-full relative overflow-hidden">
              <div className="flex flex-col h-full gap-3">
                <MultipleSelectDropdown
                  columnNames={lipid_name}
                  setSelectedColumns={handleSelectedColumns}
                  defaultValue={active_lipid.map((val) => val.name)}
                />
                <div className="mt-3 pb-16 space-y-4 overflow-y-auto">
                  {active_lipid.map((val, ind) => (
                    <div key={ind} className="flex items-center gap-2">
                      <Switch
                        size="small"
                        checked={val.active}
                        onChange={(e) => {
                          let temp = [...active_lipid];
                          temp[ind].active = !temp[ind].active;
                          setActiveLipid([...temp]);
                        }}
                      />
                      <h1 className="font-medium text-gray-800">{val.name}</h1>
                      <input
                        type="number"
                        value={active_lipid[ind].value || 0}
                        onChange={(e) => {
                          let temp = [...active_lipid];
                          temp[ind].value = e.target.value;
                          setActiveLipid([...temp]);
                        }}
                        className="w-20 ml-auto bg-transparent border-2 shadow rounded-md px-2 py-0.5 outline-none border-gray-300"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="absolute z-50 left-0 text-center bg-gray-200 shadow-xl bottom-0 border-t-2 w-full pt-4 pb-2">
              <button
                className={`${
                  loading ? "bg-blue-300" : "bg-blue-500"
                } px-4 py-2 text-white rounded mb-2 text-sm`}
                disabled={loading}
                onClick={handleCalculate}
              >
                {!loading ? (
                  "Calculate"
                ) : (
                  <CircularProgress size={20} color="primary" />
                )}
              </button>
              {prediction && (
                <p className="font-bold text-sm">Prediction: {prediction}</p>
              )}
            </div>
          </>
        )}
      </div>
      <div className="w-full h-full relative">
        {showPlot ? (
          <img src={image} alt="" className="mt-12" />
        ) : (
          <ChartComponent active_lipid={active_lipid} />
        )}

        <div className="absolute top-0 flex items-center  text-gray-700 left-1/2 py-2 -translate-x-1/2">
          <p className={`${showPlot && "font-semibold"}`}>Prediction Graph</p>
          <Switch
            size="medium"
            checked={!showPlot}
            onChange={(e) => {
              if (image) {
                setShowPlot(!e.target.checked);
              }
            }}
            disabled={loading}
          />
          <p className={`${!showPlot && "font-semibold"}`}>Molecule Graph</p>
        </div>
      </div>
    </div>
  );
}

export default App;
