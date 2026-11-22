import { build } from "esbuild";
import { copyFile, mkdir } from "node:fs/promises";

await mkdir("dist", { recursive: true });
await build({
  entryPoints: ["src/workspace.jsx"],
  outfile: "dist/workspace.js",
  bundle: true,
  minify: true,
  target: "chrome120",
  jsx: "automatic",
  loader: { ".woff2": "file" },
  assetNames: "[name]-[hash]",
  define: { "process.env.NODE_ENV": '"production"' },
});
await copyFile("index.html", "dist/index.html");
await mkdir("dist/licenses", { recursive: true });
for (const name of [
  "react", "react-dom", "@radix-ui/themes", "lucide-react",
  "@fontsource-variable/dm-sans", "@fontsource-variable/bricolage-grotesque",
]) {
  await copyFile(`node_modules/${name}/LICENSE`, `dist/licenses/${name.replace("/", "-")}.txt`);
}
