declare module "react-cytoscapejs" {
  import type { Core, ElementDefinition, Stylesheet, LayoutOptions } from "cytoscape";
  import type { CSSProperties } from "react";

  interface CytoscapeComponentProps {
    elements: ElementDefinition[];
    style?: CSSProperties;
    layout?: LayoutOptions;
    stylesheet?: Stylesheet[];
    cy?: (cy: Core) => void;
    [key: string]: unknown;
  }

  const CytoscapeComponent: React.FC<CytoscapeComponentProps>;
  export default CytoscapeComponent;
}
