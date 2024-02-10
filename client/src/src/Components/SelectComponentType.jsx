import { Select } from "antd";
import { useDispatch } from "react-redux";
import { changeNumOfComp, changeOperationID } from "../Slices/LipidSlice";

const SelectComponentType = ({ type, setType, setLipidInput }) => {
  const dispatch = useDispatch();
  return (
    <div className="space-y-1.5">
      <label htmlFor="numOfComp" className=" text-gray-700 font-medium">
        Number of Components
      </label>
      <Select
        size="large"
        defaultValue={type}
        style={{ width: "100%" }}
        onChange={(e) => {
          if (e === "single") {
            setLipidInput([{ name: "", percentage: 100 }]);
          } else {
            setLipidInput([
              { name: "", percentage: 0 },
              { name: "", percentage: 0 },
            ]);
          }
          dispatch(changeNumOfComp(e));
          dispatch(changeOperationID("0"));
          setType(e);
        }}
        options={[
          { value: "single", label: "Single" },
          { value: "multiple", label: "Multiple" },
        ]}
      />
    </div>
  );
};

export default SelectComponentType;
