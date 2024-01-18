import { Select } from "antd";

const SelectComponentType = ({ type, setType, setLipidInput }) => (
    <div className="space-y-1.5">
      <label htmlFor="numOfComp" className="text-sm text-gray-700 font-medium">
        Number of Components
      </label>
      <Select
        defaultValue={type}
        style={{ width: "100%" }}
        onChange={(e) => {
          if (e === "single") {
            setLipidInput([{ name: "", percentage: 0 }]);
          } else {
            setLipidInput([{ name: "", percentage: 0 }, { name: "", percentage: 0 }]);
          }
          setType(e);
        }}
        options={[
          { value: "single", label: "Single" },
          { value: "multiple", label: "Multiple" },
        ]}
      />
    </div>
  );
  
  export default SelectComponentType