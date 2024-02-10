import CheckBoxIcon from "@mui/icons-material/CheckBox";
import CheckBoxOutlineBlankIcon from "@mui/icons-material/CheckBoxOutlineBlank";
import { Autocomplete, TextField } from "@mui/material";
import { useEffect, useState } from "react";

const icon = <CheckBoxOutlineBlankIcon fontSize="small" />;
const checkedIcon = <CheckBoxIcon fontSize="small" />;

const MultipleSelectDropdown = ({
  columnNames,
  setSelectedColumns,
  defaultValue,
}) => {
  const [selectedItems, setSelectedItems] = useState([]);

  useEffect(() => {
    if (defaultValue) setSelectedItems(defaultValue);
  }, [defaultValue]);

  return (
    <div className="mt-1">
      <Autocomplete
        multiple
        size="small"
        id="tags-outlined"
        options={columnNames}
        value={selectedItems}
        getOptionLabel={(option) => option}
        filterSelectedOptions
        onChange={(e, newValue) => {
          setSelectedItems(newValue);
          setSelectedColumns(newValue);
        }}
        renderInput={(params) => <TextField {...params} label="Lipid Name" />}
      />
    </div>
  );
};

export default MultipleSelectDropdown;
