import logging
import os
import re
import subprocess

from moatless.verify.types import VerificationError

logger = logging.getLogger(__name__)


def run_maven_and_parse_errors(repo_dir: str) -> list[VerificationError]:
    try:
        os.environ["JAVA_HOME"] = "/home/albert/.sdkman/candidates/java/17.0.8-tem"

        result = subprocess.run(
            "./mvnw clean compile",
            cwd=repo_dir,
            check=False,
            text=True,
            shell=True,
            capture_output=True,
        )

        stdout = result.stdout
        stderr = result.stderr

        combined_output = stdout + "\n" + stderr
        return parse_compilation_errors(combined_output)

    except subprocess.CalledProcessError as e:
        logger.warning("Error running Maven command:")
        logger.warning(e.stderr)


def parse_compilation_errors(output: str) -> list[VerificationError]:
    error_pattern = re.compile(r"\[ERROR\] (.*?):\[(\d+),(\d+)\] (.*)")
    matches = error_pattern.findall(output)

    errors = []
    for match in matches:
        file_path, line, column, message = match
        error = VerificationError(
            code="COMPILATION_ERROR",
            file_path=file_path.strip(),
            message=message.strip(),
            line=int(line),
        )
        errors.append(error)
    return errors


if __name__ == "__main__":
    repo_dir = "/home/albert/repos/p24/system-configuration/modules/system-configuration-module"

    content = """[INFO] /home/albert/repos/p24/system-configuration/modules/system-configuration-module/core/src/main/java/se/alerisx/mhp/configuration/entity/OriginEntity.java: Recompile with -Xlint:deprecation for details.
[INFO] /home/albert/repos/p24/system-configuration/modules/system-configuration-module/core/src/main/java/se/alerisx/mhp/configuration/domain/impl/rule/AbstractRuleConditional.java: Some input files use unchecked or unsafe operations.
[INFO] /home/albert/repos/p24/system-configuration/modules/system-configuration-module/core/src/main/java/se/alerisx/mhp/configuration/domain/impl/rule/AbstractRuleConditional.java: Recompile with -Xlint:unchecked for details.
[INFO] -------------------------------------------------------------
[ERROR] COMPILATION ERROR : 
[INFO] -------------------------------------------------------------
[ERROR] /home/albert/repos/p24/system-configuration/modules/system-configuration-module/core/src/main/java/se/alerisx/mhp/configuration/domain/impl/CareProviderImplBuilder.java:[37,46] invalid method reference
  cannot find symbol
    symbol:   method getCountryCode()
    location: interface se.alerisx.mhp.configuration.domain.Origin
[ERROR] /home/albert/repos/p24/system-configuration/modules/system-configuration-module/core/src/main/java/se/alerisx/mhp/configuration/domain/impl/SystemOriginImpl.java:[154,5] method does not override or implement a method from a supertype
[ERROR] /home/albert/repos/p24/system-configuration/modules/system-configuration-module/core/src/main/java/se/alerisx/mhp/configuration/domain/impl/OriginImpl.java:[486,26] cannot find symbol
  symbol:   method getCountryCode()
  location: variable parent of type se.alerisx.mhp.configuration.domain.Origin
[INFO] 3 errors 
[INFO] -------------------------------------------------------------
[INFO] ------------------------------------------------------------------------
[INFO] Reactor Summary for system-configuration 0.0.0-SNAPSHOT:
[INFO] 
[INFO] system-configuration ............................... SUCCESS [  2.283 s]
[INFO] system-configuration-core .......................... FAILURE [ 16.507 s]
[INFO] rule-engine ........................................ SKIPPED
[INFO] system-configuration-object-storage ................ SKIPPED
[INFO] system-configuration-cli ........................... SKIPPED
"""

    errors = parse_errors(content)

    logging.basicConfig(level=logging.INFO)
    # errors = run_maven_and_parse_errors(repo_dir)
    print(errors)
    for error in errors:
        print(error)
