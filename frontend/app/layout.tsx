import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "AxiomESG",
  description: "ESG intelligence layer for unstructured documents",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
